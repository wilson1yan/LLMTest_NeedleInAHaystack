import argparse
import itertools
from tqdm import tqdm
import functools
from pathlib import Path
from dotenv import load_dotenv
import os
import tiktoken
import torch
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer #LlamaForCausalLM, LlamaTokenizerFast
import transformers
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()


def read_files(directory):
    context = ""
    for file in glob.glob(directory):
        with open(file, 'r') as f:
            context += f.read()
    return context

def encode_and_trim(context, context_length, enc):
    tokens = enc.encode(context)
    if len(tokens) > context_length:
        context = enc.decode(tokens[:context_length])
    return context

def insert_needle(needle, context, depth_percent, context_length, enc):
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 150

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

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

def generate_context(needle, context_length, depth_percent):
    # Load up tiktoken so we navigate tokens more easily
    enc = tiktoken.encoding_for_model("gpt-4-1106-preview")

    # Get your Paul Graham files loaded into a string
    context = read_files("PaulGrahamEssays/*.txt")

    # Truncate the Paul Graham essays to the context length you desire
    context = encode_and_trim(context, context_length, enc)

    # Insert your random statement according to your depth percent
    context = insert_needle(needle, context, depth_percent, context_length, enc)

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


class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False


class Sampler:
    def __init__(self, ckpt_path, device):
        self.ckpt_path = ckpt_path
        self.model = AutoModelForCausalLM.from_pretrained(ckpt_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        self.tokenizer.add_bos_token = False
        self.device = device

    def __call__(self, context):
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
            prompt = f"{system_prompt} USER: {context}\nWhat is the most fun thing to do in San Francisco based on the context? Don't give information outside the document or repeat your findings. Keep your response short and direct. Assistant: "

        inputs = self.tokenizer(prompt, return_tensors="pt")
        starting_idx = inputs.input_ids.shape[-1]
        stopping_criteria_list = transformers.StoppingCriteriaList([
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=self.tokenizer(
                    "</s>",
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to("cuda"),
                starting_idx=starting_idx)
        ])
        output = self.model.generate(
            inputs.input_ids.to(self.device),
            do_sample=False,
            max_new_tokens=250,
            stopping_criteria=stopping_criteria_list,
            early_stopping=True
        )
        output = output[0, starting_idx:]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        return AIMessage(content=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('--no_gpt_4', action='store_true')
    parser.add_argument('--max_context', type=int, default=128000)
    parser.add_argument('--min_context', type=int, default=1000)
    parser.add_argument('--num_bins_context', type=int, default=15)
    parser.add_argument('--num_bins_depth', type=int, default=15)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output_folder', type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda')

    needle = """
    The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
    """
    question_to_ask = "What is the most fun thing to do in San Francisco?"

    # The code will check to see if a context_length, depth percent and version number have already been checked yet
    # Change the version # if you would like to run the results multiple times.
    # If you're just testing, then leave as version=1
    results_version = args.seed

    # This will produce a list of context lengths for each experiment iteration. Make sure the max context length is within the bounds of your models limits.
    context_lengths = np.round(np.linspace(args.min_context, args.max_context, num=args.num_bins_context, endpoint=True)).astype(int)

    # This will product a list of document depths to place your random statement (needle) at.
    # Suggestion: Try out different distributions (like a sigmoid) to test non-evenly space intervals
    document_depth_percents = np.round(np.linspace(0, 100, num=args.num_bins_depth, endpoint=True)).astype(int)

    # The model we are testing. As of now it's set up for chat models with OpenAI
    #model_to_test = ChatOpenAI(model='gpt-4', temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey'))
    model_to_test = Sampler(args.ckpt, device)

    # This will get logged on your results
    model_to_test_description = args.ckpt

    evaluation_model  = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey'))

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{Path(args.ckpt).stem}_{args.rank}.json"

    # Run through each iteration of context_lengths and depths
    cfgs = list(itertools.product(document_depth_percents, context_lengths))
    cfgs = np.array_split(cfgs, args.size)[args.rank].tolist()
    for depth_percent, context_length in tqdm(cfgs):
        # Load results from file. 
        try:
            with output_file.open('r') as f:
                results = json.load(f)
        except FileNotFoundError:
            results = []
            pass

        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if result_exists(results, context_length, depth_percent, results_version, model_to_test_description):
            continue

        # Go generate the required length context and place your needle statement in
        context = generate_context(needle, context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        #messages = [
        #    SystemMessage(
        #        content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
        #    ),
        #    HumanMessage(
        #        # This is the PG essays with your needle/random statement placed in it
        #        # This is your haystack with a needle placed in it.
        #        content=context
        #    ),
        #    HumanMessage(
        #        # This is the question you'll ask to the model to tr≠≠y and retrieve your random statement/needle.
        #        content="What is the most fun thing to do in San Francico based on the context? Don't give information outside the document or repeat your findings"
        #    ),
        #]

        # Go see if the model can answer the question to pull out your random fact
        response = model_to_test(context)
        if args.no_gpt_4:
            print(f"Context Length: {context_length}, Depth Percent: {depth_percent}")
            print(f"Response: {response.content}\nAnswer: {needle}")
            continue

        # Compare the reponse to the actual needle you placed
        score = evaluate_response(response, needle, question_to_ask, evaluation_model)

        results.append({
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : int(depth_percent),
            'version' : results_version,
            'needle' : needle,
            'model_response' : response.content,
            'score' : score
        })

        print (f"Result #: {len(results)}/{len(context_lengths) * len(document_depth_percents)}")
        print (f"Context: {context_length} tokens")
        print (f"Depth: {depth_percent}%")
        print (f"Score: {score}")
        print (f"Response: {response.content}\n")

        # Save results to a JSON file each run
        with output_file.open('w') as f:
            json.dump(results, f)

        # Optional. Sleep for a bit to stay under the rate limit
        # Rate limit is 150K tokens/min so it's set at 120K for some cushion
        #sleep_time = (context_length / 120000)*60
        # print (f"Sleeping: {sleep_time}\n")
        #time.sleep(sleep_time)
