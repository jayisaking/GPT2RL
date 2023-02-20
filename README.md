# GPT2RL
## Models Download Link

| Model Name | URL |
| ---------- | --- |
| Whole Model (Including GPT2 and Q in Q-Learning) | [model]("https.gg")|
| GPT2 | [model]("https.gg")|

##Training Step
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

##Testing Step
```
python3 test.py 
--beam [Generate beam sentences at a time] 
--root_path [Whole testing process and saving will base on this path]
--dataset_folder [The name of dataset folder in root_path]
--gpu [GPU name]
--model_path [Checkpoint path for whole model (Including GPT2 and Q in Q-Learning)]
--pretrained_gpt [GPT2 pretrained path]
```