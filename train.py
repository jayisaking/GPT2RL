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
    parser = argparse.ArgumentParser('Set parameters for training GPT by RL', add_help = False)
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--batch_size', default = 4, type=int)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--epochs', default = 32, type = int)
    parser.add_argument('--lr_drop', default = 12, type = int)
    parser.add_argument('--eval_freq', default = 3, type = int)

    # * Loss coefficients
    parser.add_argument('--Q_discounter_factor', default = 0.9, type = float)

    parser.add_argument('--kl_control', default = 1, type = float,
                        help = "coefficent of KL loss")
    parser.add_argument('--gpt_loss_coefficient', default = 1, type = float,
                        help = "coefficent of GPT loss")

    # dataset parameters
    parser.add_argument('--dataset_folder', default = 'dataset')
    parser.add_argument('--root_path', default = os.getcwd())
    parser.add_argument('--gpu', default = 'cuda')

    return parser
def get_output_dir(pwd):
    runs_dir = os.path.join(pwd, 'runs')
    train_dir = os.path.join(runs_dir, 'train')
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    counter = 1
    exp = f"exp{counter}"
    while exp in os.listdir(train_dir):
        counter += 1
        exp = f"exp{counter}"
    output_dir = os.path.join(train_dir, exp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'models')):
        os.mkdir(os.path.join(output_dir, 'models'))
    with open(os.path.join(output_dir, 'log.csv'), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'KLLoss', 'GPTLoss', 'QLoss', 'TOTALLoss'])
    return output_dir
def main(args):
    print(args)
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    lr_drop = args.lr_drop
    Q_discounter_factor = args.Q_discounter_factor
    kl_control = args.kl_control
    gpt_loss_coefficient = args.gpt_loss_coefficient
    pwd = args.root_path
    data_root = args.dataset_folder
    output_dir = get_output_dir(pwd)
    device = args.gpu
    gpt2 = GPT2LMHeadModel.from_pretrained(os.path.join(pwd, 'GPT-2/GPT2_finetune_2'))
    gpt2_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    gpt_wrapper = GPT2Wrapper(gpt2, tokenizer = gpt2_tokenizer, device = device)
    Q_A = Q(gpt_tokenizer = gpt2_tokenizer, gamma = Q_discounter_factor, bert_name = 'bert-base-chinese', device = device)
    Q_B = Q(gpt_tokenizer = gpt2_tokenizer, gamma = Q_discounter_factor, bert_name = 'bert-base-chinese', device = device)
    toxic_words, non_sense_response = GPT2DataSet.get_toxic_ids_and_non_sense_response(gpt2_tokenizer)
    R = Reward(gpt = gpt2, question_mark_token = 136, toxic_words = toxic_words, gpt_tokenizer = gpt2_tokenizer,
                non_sense_response = non_sense_response, eos_token = 102, device = device, bos_token = 101)
    criterion = nn.MSELoss()
    gpt_wrapper.to(device)
    Q_A.to(device)
    Q_B.to(device)
    optimizer = torch.optim.Adam([{ 'params': [p for p in gpt_wrapper.parameters() if p.requires_grad]},
                                { 'params': [p for p in Q_A.parameters() if p.requires_grad]},
                                { 'params': [p for p in Q_B.parameters() if p.requires_grad]},
                                ], lr = lr, weight_decay = weight_decay)
    print('total parameters: ', sum([p.numel() for p in gpt_wrapper.parameters() if p.requires_grad]) + 
        sum([p.numel() for p in Q_A.parameters() if p.requires_grad]) + 
        sum([p.numel() for p in Q_B.parameters() if p.requires_grad]))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
    train_dataset = GPT2DataSet(tokenizer = gpt2_tokenizer, max_len = 256, root_path = pwd, status = 'train', dataset_root_path = data_root)
    test_dataset = GPT2DataSet(tokenizer = gpt2_tokenizer, max_len = 256, root_path = pwd, status = 'test', dataset_root_path = data_root)
    q_losses = []
    gpt_losses = []
    kl_losses = []
    total_losses = []
    print('Start training')
    for epoch in range(epochs):
        q_loss, kl_loss, gpt_loss, total_loss = train_one_epoch(epoch = epoch, gpt = gpt_wrapper, Q_A = Q_A, Q_B = Q_B,
                                                                optimizer = optimizer, R = R, dataset = train_dataset, device = device, batch_size = batch_size,
                                                                max_len = 256, beam = 3, update_time_per_episode = 10, criterion = criterion, 
                                                                kl_control = kl_control, gpt_loss_coefficient = gpt_loss_coefficient)
        q_losses.append(float(q_loss))
        kl_losses.append(float(kl_loss))
        gpt_losses.append(float(gpt_loss))
        total_losses.append(float(total_loss))
        torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/latest.pth'))
        if min(q_losses) == q_loss:
            time.sleep(0.1)
            torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/best_q_loss.pth'))
        if min(kl_losses) == kl_loss:
            time.sleep(0.1)
            torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/best_kl_loss.pth'))
        if min(gpt_losses) == gpt_loss:
            time.sleep(0.1)
            torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/best_gpt_loss.pth'))
        if min(total_losses) == total_loss:
            time.sleep(0.1)
            torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/best_total_loss.pth'))
        with open(os.path.join(output_dir, 'log.csv'), 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, kl_loss, gpt_loss, q_loss, total_loss])
        if epoch % args.eval_freq == 0:
            q_loss, kl_loss, gpt_loss, total_loss = evaluate(gpt = gpt_wrapper, Q_A = Q_A, Q_B = Q_B, R = R, dataset = test_dataset, device = device, batch_size = batch_size,
                                                                max_len = 256, beam = 3, criterion = criterion, 
                                                                kl_control = kl_control, gpt_loss_coefficient = gpt_loss_coefficient)

            # print the evaluation results
            print('=======================================test=======================================')
            print(f"QLoss : {q_loss} KLLoss {kl_loss}: GPTLoss : {gpt_loss} TotalLoss : {total_loss}")
            print('=======================================test=======================================')

        R.update_model(gpt_wrapper.gpt)
            
def train_one_epoch(epoch: int, gpt: GPT2Wrapper, Q_A: Q, Q_B: Q,  optimizer: torch.optim.Optimizer = None, R: Reward = None,
                    dataset: Iterable = GPT2DataSet(), device: torch.device = 'cpu', batch_size = 4, max_len = 256, beam = 3, update_time_per_episode = 10, criterion = nn.MSELoss(), kl_control = 0.01,
                    gpt_loss_coefficient = 0.1):
    gpt.train()
    # two Qs for Double DQN
    Q_A.train()
    Q_B.train()
    gpt.to(device)
    Q_A.to(device)
    Q_B.to(device)
    kl_losses = []
    gpt_losses = []
    q_losses = []
    total_losses = []
    

    with tqdm(total = len(dataset) // batch_size) as t:
        for step in range(len(dataset) // batch_size):
            t.set_description(f"Epoch {epoch}")

            prev_utterance = []
            response = []
            for mini_step in range(step * batch_size, (step + 1) * batch_size):
                pair = dataset[mini_step]
                prev_utterance.append(pair['prev_utterance'].to(device))
                response.append(pair['prev_utterance'].to(device))
            utter_length = None
            generate_time = 0
            with torch.no_grad(): # generate episode
                while (utter_length is None or all(utter_length < max_len)) and generate_time < 4:
                    generate_time += 1
                    if utter_length is None:
                        rslt, msk, sco, smlgprbs, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance = prev_utterance, response = response, beam = beam, max_len = max_len,
                                                                device = device)
                        # soft sampling
                        previous_Q_distribution= F.softmax((Q_A(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False) + 
                        Q_B(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False)).view(batch_size, -1), dim = -1)
                        select = torch.multinomial(previous_Q_distribution, 1)
                        previous_result = rslt.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        results = previous_result.detach().clone()
                        masks = msk.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        scores = sco.gather(index = select, dim = 1).squeeze(1)
                        utter_length = utrlen.gather(index = select, dim = 1).squeeze(1)
                        sum_logprobs = smlgprbs.gather(index = select, dim = 1).squeeze(1)
                        results_prev_utterance = [rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                        results_response = [rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                    else:
                        rslt, msk, sco, smlgprbs, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance = previous_result, beam = beam, max_len = max_len,
                                                            device = device)

                        previous_Q_distribution = F.softmax((Q_A(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False) + 
                        Q_B(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False)).view(batch_size, -1), dim = -1)
                        select = torch.multinomial(previous_Q_distribution, 1)
                        previous_result = rslt.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        
                        results = torch.cat((results, previous_result.detach().clone()), dim = 0)
                        masks = torch.cat((masks, msk.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)), dim = 0)
                        scores = torch.cat((scores, sco.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        utter_length = torch.cat((utter_length, utrlen.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        sum_logprobs = torch.cat((sum_logprobs, smlgprbs.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        results_prev_utterance.extend([rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                        results_response.extend([rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                reward = R([{'prev_utterance': utt, 'response': res} for utt, res in zip(results_prev_utterance[batch_size: ], results_response[batch_size: ])])
                results_Q, mask_Q = Q_A.get_processed(prev_utterance = results, bert_tokens = False, max_len = max_len * 2)
            for update_time in range(update_time_per_episode):
                if random.random() >= 0.5: # update Q_A
                    q_estimate = Q_A.forward(prev_utterance = results_Q[: len(results_Q) - batch_size], response = None, mask = mask_Q[: len(results_Q) - batch_size], bert_tokens = True, max_len = max_len * 2, processed = True)
                    q_target = reward + Q_A.gamma * Q_B.forward(prev_utterance = results_Q[batch_size: ], response = None, mask = mask_Q[batch_size: ], bert_tokens = True, max_len = max_len * 2, processed = True)
                else: # update Q_B
                    q_estimate = Q_B.forward(prev_utterance = results_Q[: len(results_Q) - batch_size], response = None, mask = mask_Q[: len(results_Q) - batch_size], bert_tokens = True, max_len = max_len * 2, processed = True)
                    q_target = reward + Q_B.gamma * Q_A.forward(prev_utterance = results_Q[batch_size: ], response = None, mask = mask_Q[batch_size: ], bert_tokens = True, max_len = max_len * 2, processed = True)
                q_loss = criterion(q_estimate, q_target) 
                kl_loss = - kl_control * torch.log(torch.mean(q_estimate.exp()))
                probs = gpt.get_prob(results[: len(results_Q) - batch_size], masks[: len(results_Q) - batch_size], results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])
                gpt_loss = torch.sum(probs.detach().clone().exp() * probs * q_target)
                total_loss = q_loss + kl_control * kl_loss + gpt_loss_coefficient * gpt_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                kl_losses.append(float(kl_loss.detach().clone()))
                gpt_losses.append(float(gpt_loss.detach().clone()))
                q_losses.append(float(q_loss.detach().clone()))
                total_losses.append(float(total_loss.detach().clone()))
                t.set_postfix(klloss = float(kl_losses[-1]), gpt_loss = float(gpt_losses[-1]),
                          q_loss = float(q_losses[-1]), total_loss = float(total_losses[-1]))
            t.update(1)
    return np.mean(q_losses), np.mean(kl_losses), np.mean(gpt_losses), np.mean(total_losses)
@torch.no_grad()
def evaluate(gpt, Q_A, Q_B, R, dataset, device, batch_size = 4,max_len = 256, beam = 3, criterion = nn.MSELoss(), kl_control = 0.1, gpt_loss_coefficient = 0.1):
    gpt.eval()
    # two Qs for Double DQN
    Q_A.eval()
    Q_B.eval()
    gpt.to(device)
    Q_A.to(device)
    Q_B.to(device)
    kl_losses = []
    gpt_losses = []
    q_losses = []
    total_losses = []
    

    with tqdm(total = len(dataset) // batch_size) as t:
        for step in range(len(dataset) // batch_size):
            t.set_description(f"Validation")

            prev_utterance = []
            response = []
            for mini_step in range(step * batch_size, (step + 1) * batch_size):
                pair = dataset[mini_step]
                prev_utterance.append(pair['prev_utterance'].to(device))
                response.append(pair['prev_utterance'].to(device))
            utter_length = None
            generate_time = 0
            with torch.no_grad(): # generate episode
                while (utter_length is None or all(utter_length < max_len) or generate_time < 2) and generate_time < 4:
                    generate_time += 1
                    if utter_length is None:
                        rslt, msk, sco, smlgprbs, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance = prev_utterance, response = response, beam = beam, max_len = max_len,
                                                                device = device)
                        # soft sampling
                        previous_Q_distribution= F.softmax((Q_A(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False) + 
                        Q_B(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False)).view(batch_size, -1), dim = -1)
                        select = torch.multinomial(previous_Q_distribution, 1)
                        previous_result = rslt.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        results = previous_result.detach().clone()
                        masks = msk.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        scores = sco.gather(index = select, dim = 1).squeeze(1)
                        utter_length = utrlen.gather(index = select, dim = 1).squeeze(1)
                        sum_logprobs = smlgprbs.gather(index = select, dim = 1).squeeze(1)
                        results_prev_utterance = [rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                        results_response = [rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                    else:
                        rslt, msk, sco, smlgprbs, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance = previous_result, beam = beam, max_len = max_len,
                                                            device = device)

                        previous_Q_distribution = F.softmax((Q_A(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False) + 
                        Q_B(prev_utterance = rslt.view(batch_size * beam, -1), response = None, mask = None, bert_tokens = False, max_len = max_len * 2, processed = False)).view(batch_size, -1), dim = -1)
                        select = torch.multinomial(previous_Q_distribution, 1)
                        previous_result = rslt.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)
                        
                        results = torch.cat((results, previous_result.detach().clone()), dim = 0)
                        masks = torch.cat((masks, msk.gather(index = select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim = 1).squeeze(1)), dim = 0)
                        scores = torch.cat((scores, sco.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        utter_length = torch.cat((utter_length, utrlen.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        sum_logprobs = torch.cat((sum_logprobs, smlgprbs.gather(index = select, dim = 1).squeeze(1)), dim = 0)
                        results_prev_utterance.extend([rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                        results_response.extend([rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                reward = R([{'prev_utterance': utt, 'response': res} for utt, res in zip(results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])])
                results_Q, mask_Q = Q_A.get_processed(prev_utterance = results, bert_tokens = False, max_len = max_len * 2)
            print(results_Q.shape, mask_Q.shape)
            if random.random() >= 0.5:
                    q_estimate = Q_A.forward(prev_utterance = results_Q[: len(results_Q) - batch_size], response = None, mask = mask_Q[: len(results_Q) - batch_size], bert_tokens = True, max_len = max_len * 2, processed = True)
                    q_target = reward + Q_A.gamma * Q_B.forward(prev_utterance = results_Q[batch_size: ], response = None, mask = mask_Q[batch_size: ], bert_tokens = True, max_len = max_len * 2, processed = True)
            else: # update Q_B
                    q_estimate = Q_B.forward(prev_utterance = results_Q[: len(results_Q) - batch_size], response = None, mask = mask_Q[: len(results_Q) - batch_size], bert_tokens = True, max_len = max_len * 2, processed = True)
                    q_target = reward + Q_B.gamma * Q_A.forward(prev_utterance = results_Q[batch_size: ], response = None, mask = mask_Q[batch_size: ], bert_tokens = True, max_len = max_len * 2, processed = True)
            q_loss = criterion(q_estimate, q_target) 
            kl_loss = - kl_control * torch.log(torch.mean(q_estimate.exp()))
            probs = gpt.get_prob(results[: len(results_Q) - batch_size], masks[: len(results_Q) - batch_size], results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])
            gpt_loss = torch.sum(probs.detach().clone().exp() * probs * q_target)
            total_loss = q_loss + kl_control * kl_loss + gpt_loss_coefficient * gpt_loss
                
            kl_losses.append(float(kl_loss.detach().clone()))
            gpt_losses.append(float(gpt_loss.detach().clone()))
            q_losses.append(float(q_loss.detach().clone()))
            total_losses.append(float(total_loss.detach().clone()))
            t.set_postfix(klloss = float(kl_losses[-1]), gpt_loss = float(gpt_losses[-1]),
                          q_loss = float(q_losses[-1]), total_loss = float(total_losses[-1]))
            t.update(1)
    return np.mean(q_losses), np.mean(kl_losses), np.mean(gpt_losses), np.mean(total_losses)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)