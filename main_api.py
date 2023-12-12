import argparse
import itertools
from tqdm import tqdm
import functools
import multiprocessing as mp
from pathlib import Path
import openai
from dotenv import load_dotenv
import os
import tiktoken
import torch
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

tokens_cache = None
needle = """
The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
"""
question_to_ask = "What is the most fun thing to do in San Francisco?"


def read_files(directory):
    context = ""
    for file in glob.glob(directory):
        with open(file, 'r') as f:
            context += f.read()
    return context

def insert_needle(needle, context, depth_percent, context_length, enc):
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 150

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent >= 1.0:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * depth_percent)

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = enc.encode('.')
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = enc.decode(tokens_new_context)
    return new_context

def generate_context(needle, context_length, depth_percent, llama_tokenizer):
    # Load up tiktoken so we navigate tokens more easily
    enc = tiktoken.encoding_for_model("gpt-4-1106-preview")

    # Get your Paul Graham files loaded into a string
    context = read_files("PaulGrahamEssays/doc.txt")

    # Truncate the Paul Graham essays to the context length you desire
    global tokens_cache # use LlamaTokenizer for this part
    if tokens_cache is None:
        #tokens_cache = llama_tokenizer.encode(context)
        tokens_cache = json.load(open('tokens_cache.json', 'r'))
    print(len(tokens_cache))
    context = llama_tokenizer.decode(tokens_cache[:context_length])

    # Insert your random statement according to your depth percent
    context = insert_needle(needle, context, depth_percent, context_length, enc) # USE tiktoken as LLAMA doesn't do single period encoding well
    idx = context.find(needle)
    print(len(context) / context_length, context_length, idx / len(context), depth_percent)

    return context

def evaluate_response(response, needle, question_to_ask, evaluation_model):
    accuracy_criteria = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Keep your explanations extremely short, just give the score
        """
    }

    # Using GPT-4 to evaluate
    evaluator = load_evaluator(
        "labeled_score_string",
        criteria=accuracy_criteria,
        llm=evaluation_model,
    )

    eval_result = evaluator.evaluate_strings(
        # The models response
        prediction=response,

        # The actual answer
        reference=needle,

        # The question asked
        input=question_to_ask,
    )

    return int(eval_result['score'])

def result_exists(results, context_length, depth_percent, version, model):
    """
    Checks to see if a result has already been evaluated or not
    """
    conditions_met = []
    for result in results:
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == version
        model_met = result['model'] == model
        conditions_met.append(context_length_met and depth_percent_met and version_met)
    return any(conditions_met)


class Sampler:
    def __init__(self, ckpt_path, device):
        self.ckpt_path = ckpt_path
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        self.tokenizer.add_bos_token = False

    def __call__(self, context):
        #openai.api_key = "EMPTY"
        old_api_base = openai.api_base
        openai.api_base = "http://localhost:8000/v1"

        if 'vicuna' in self.ckpt_path or 'longchat' in self.ckpt_path:
            system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"{system_prompt} USER: {context}\nWhat is the most fun thing to do in San Francisco based on the context? Don't give information outside the document or repeat your findings. Keep your response short and direct. Assistant: "
        elif 'Mistral' in self.ckpt_path:
            messages = [
               {"role": "user", "content": f"{context}\nWhat is the most fun thing to do in San Francisco based on the context? Don't give information outside the document or repeat your findings. Keep your response short and direct."},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            system_prompt = "You are a helpful assistant."
            prompt = f"{system_prompt} USER: {context} What is the most fun thing to do in San Francisco based on the context? Don't give information outside the document or repeat your findings. Keep your response short and direct. Assistant: "

        stop = '</s>'

        n_retries = 0
        while True:
            try:
                output = openai.Completion.create(
                    model=os.path.basename(self.ckpt_path),
                    prompt=prompt,
                    max_tokens=250,
                    stop=stop,
                    temperature=0.,
                ).choices[0].text
                break
            except:
                n_retries += 1
                print(f"retry {n_retries}")

        openai.api_base = old_api_base
        return AIMessage(content=output)


def eval_example(inp, model_to_test):
    depth_percent, context_length = inp
    context_length = int(context_length)

    # Go generate the required length context and place your needle statement in
    context = generate_context(needle, context_length, depth_percent, model_to_test.tokenizer)

    # Go see if the model can answer the question to pull out your random fact
    response = model_to_test(context)
    if args.no_gpt_4:
        return {
            'context_length': context_length,
            'depth_percent': depth_percent,
            'model_response': response.content,
            'needle': needle
        }


    content = response.content
    if "Dolores" in content and "sandwich" in content:
        score = 10
    else:
        score = 3

    # Compare the reponse to the actual needle you placed
    #score = evaluate_response(response, needle, question_to_ask, evaluation_model)


    return {
        'model' : model_to_test_description,
        'context_length' : int(context_length),
        'depth_percent' : depth_percent,
        'version' : results_version,
        'needle' : needle,
        'model_response' : response.content,
        'score' : score
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('--no_gpt_4', action='store_true')
    parser.add_argument('--max_context', type=int, default=128000)
    parser.add_argument('--min_context', type=int, default=1000)
    parser.add_argument('--num_bins_context', type=int, default=15)
    parser.add_argument('--num_bins_depth', type=int, default=15)
    parser.add_argument('--dist', type=str, default='linear', choices=['linear', 'sigmoid'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('-o', '--output_folder', type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda')

    results_version = args.seed
    context_lengths = np.round(np.linspace(args.min_context, args.max_context, num=args.num_bins_context, endpoint=True)).astype(int)

    if args.dist == 'linear':
        document_depth_percents = np.linspace(0, 1.0, num=args.num_bins_depth, endpoint=True)
    elif args.dist == 'sigmoid':
        assert args.num_bins_depth % 2 == 1
        document_depth_percents_half1 = 1 / (1 + np.exp(-np.linspace(-4.5, 0, (args.num_bins_depth - 2) // 2 + 1)))
        document_depth_percents_half2 = 1 / (1 + np.exp(-np.linspace(0, 4.5, (args.num_bins_depth - 2) // 2 + 1)))[1:]
        document_depth_percents = np.concatenate(([0], document_depth_percents_half1, document_depth_percents_half2, [1]))
    else:
        raise Exception(args.dist)

    model_to_test = Sampler(args.ckpt, device)
    model_to_test_description = args.ckpt

    evaluation_model  = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey'))

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    fname = f"{Path(args.ckpt).stem}"
    if args.suffix is not None:
        fname = f"{fname}_{args.suffix}"
    fname = f"{fname}.json"
    output_file = output_folder / fname
    print(output_file)

    cfgs = list(itertools.product(document_depth_percents, context_lengths))
    if output_file.exists():
        results = json.load(output_file.open('r'))
    else:
        results = []
    cfgs = [cfg for cfg in cfgs if not result_exists(results, cfg[1], cfg[0], results_version, model_to_test_description)]
    pool = mp.Pool(args.num_workers)
    for result in tqdm(pool.imap_unordered(functools.partial(eval_example, model_to_test=model_to_test), cfgs), total=len(cfgs)):
        if args.no_gpt_4:
            print(f"Context Length: {result['context_length']}, Depth Percent: {result['depth_percent'] * 100:.2f}%")
            print(f"Response: {result['model_response']}\nAnswer: {result['needle']}")
            continue
        print (f"Result #: {len(results)}/{len(cfgs)}")
        print (f"Context: {result['context_length']} tokens")
        print (f"Depth: {result['depth_percent'] * 100:.2f}%")
        print (f"Score: {result['score']}")
        print (f"Response: {result['model_response']}\n")
        results.append(result)
        with output_file.open('w') as f:
            json.dump(results, f)
    print('done')
