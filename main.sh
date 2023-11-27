#!/bin/bash

for i in {0..7}
do
	#CUDA_VISIBLE_DEVICES=$i python main.py --ckpt /home/wilsonyan/checkpoints/converted/7b-books3-32k-theta1M-run1122-1-ultrachatft/ --max_context 32000 --num_bins_context 4 --num_bins_depth 8 --rank $i --size 8 &
	#CUDA_VISIBLE_DEVICES=$i python main.py --ckpt lmsys/longchat-7b-v1.5-32k --max_context 32000 --num_bins_context 4 --num_bins_depth 8 --rank $i --size 8 &
	#CUDA_VISIBLE_DEVICES=$i python main.py --ckpt lmsys/vicuna-7b-v1.5-16k --max_context 32000 --num_bins_context 4 --num_bins_depth 8 --rank $i --size 8 &
	#CUDA_VISIBLE_DEVICES=$i python main.py --ckpt mistralai/Mistral-7B-Instruct-v0.1 --max_context 32000 --num_bins_context 4 --num_bins_depth 8 --rank $i --size 8 &
	CUDA_VISIBLE_DEVICES=$i python main.py --ckpt /home/wilsonyan/checkpoints/converted/7b-books3-128k-theta10M-run1122-1-ultrachatft/ --max_context 128000 --num_bins_context 10 --num_bins_depth 10 --rank $i --size 8 &
done
