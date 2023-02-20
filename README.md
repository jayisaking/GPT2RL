# GPT2RL
## Intro
This repo contains training and testing code of a conversation model.
## Models Download Link

| Model Name | URL |
| ---------- | --- |
| Whole Model (Including GPT2 and Q in Q-Learning) | [model]("https.gg")|
| GPT2 | [model]("https.gg")|

## Training Step
```
python3 train.py 
--lr [learning rate] 
--batch_size [batch_size] --epochs [epochs]
--weight_decay [weight decay in Adam] 
--lr_drop [lr scheduler alter lr for every lr_drop epochs]
--eval_freq [run validation step every eval_freq epochs] 
--Q_discounter_factor [Q-Learning Discount Factor]
--kl_control [KL Divergence Loss propotion in total loss] 
--gpt_loss_coefficient [GPT Loss propotion in total loss]
--root_path [Whole training process and saving will base on this path]
--dataset_folder [The name of dataset folder in root_path]
--gpu [GPU name]
--checkpoint [Checkpoint path for whole model (Including GPT2 and Q in Q-Learning)]
--pretrained_gpt [GPT2 pretrained path]
--max_len [Max length for sentences]
```

## Testing Step
```
python3 test.py 
--beam [Generate beam sentences at a time] 
--root_path [Whole testing process and saving will base on this path]
--dataset_folder [The name of dataset folder in root_path]
--gpu [GPU name]
--model_path [Checkpoint path for whole model (Including GPT2 and Q in Q-Learning)]
--pretrained_gpt [GPT2 pretrained path]
```

### Example Output
```
BLEU2: 0.06003492132617433 BLEU4: 0.0050537417464194614 Acc: 0.04796496030694967 Perplexity: 2.795163154602051 
Enter String To Converse, Enter quit to quit : 你今天早上吃過了嗎？[開心]
inputed string : 你今天早上吃過了嗎？[開心]
Candidate Sentences
我吃過了 0.8399896621704102
沒有，我是說，你今天早上起來吃的什麼啊 0.8241734504699707
呵 0.6519677042961121
Selected index : 1
Sentence : 沒有，我是說，你今天早上起來吃的什麼啊
```