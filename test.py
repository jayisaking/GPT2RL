from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizerFast, AutoModel, GPT2Tokenizer, BertTokenizer, BertModel, BertLMHeadModel
import json
from torch import nn
import torch
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import random
import numpy as np
from typing import Iterable
from tqdm import tqdm
import csv
import argparse
from models import *
from Data_Reward import *
import time
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for testing GPT by RL', add_help = False)
    parser.add_argument('--root_path', default = os.getcwd())
    parser.add_argument('--model_path', default = "./train/runs/exp1/models/latest.pth")
    parser.add_argument('--gpu', default = 'cuda')
    parser.add_argument('--beam', default = 3, type = int)
    return parser
def conversation(input_string, gpt_wrapper, Q_A, Q_B, gpt_tokenizer, max_len = 256, device = 'cpu', beam = 3, emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它']):
    if input_string[-1] != ']':
        input_string += f"[{emotion[random.randint(0, 5)]}]"
    input_ids = gpt_tokenizer.encode(input_string, return_tensors = 'pt')
    rslt, msk, sco, smlgprbs, utrlen, rsltprvrnce, rsltrspse = gpt_wrapper(prev_utterance = [input_ids[0]], max_len = max_len, require_grad = False, device = device, beam = beam)
    scores = (Q_A(prev_utterance = rslt[0]) + Q_B(prev_utterance = rslt[0])).exp()
    select = int(torch.multinomial(scores / sum(scores), 1))
    return rsltrspse[0], select, rslt[0][select][rslt[0][select] > 0], scores
def main(args, emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它']):
    device = args.gpu
    beam = args.beam
    pwd = args.root_path
    gpt2 = GPT2LMHeadModel.from_pretrained(os.path.join(pwd, 'GPT-2/GPT2_finetune_2'))
    gpt2_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    gpt_wrapper = GPT2Wrapper(gpt2, tokenizer = gpt2_tokenizer, device = device)
    Q_A = Q(gpt_tokenizer = gpt2_tokenizer, bert_name = 'bert-base-chinese', device = device)
    Q_B = Q(gpt_tokenizer = gpt2_tokenizer, bert_name = 'bert-base-chinese', device = device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path)
        gpt_wrapper.load_state_dict(ckpt['GPT'])
        Q_A.load_state_dict(ckpt['Q_A'])
        Q_B.load_state_dict(ckpt['Q_B'])
    else:
        print("model file doesn't exist")
    input_string = input("Enter String To Converse, Enter quit to quit : ")
    prev_utterance = ""
    while input_string != "quit":
        print("inputed string :", input_string)
        prev_utterance += input_string
        responses, selected_index, prev_utterance, scores = conversation(prev_utterance, gpt_wrapper, Q_A, Q_B, gpt2_tokenizer, beam = beam, emotion = emotion)
        print("Candidate Sentences")
        for i in range(beam):
            print(gpt2_tokenizer.decode(responses[i], skip_special_tokens = True).replace(" ", ""), float(scores[i]))
        print(f"Selected index : {selected_index}\nSentence : {gpt2_tokenizer.decode(responses[selected_index], skip_special_tokens = True).replace(' ', '')}")
        prev_utterance = gpt2_tokenizer.decode(prev_utterance, skip_special_tokens = True).replace(" ", "")
        if prev_utterance[-1] != ']':
            prev_utterance += f"[{emotion[random.randint(0, 5)]}]"
        input_string = input("Enter String To Converse, Enter quit to quit : ")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPT testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)