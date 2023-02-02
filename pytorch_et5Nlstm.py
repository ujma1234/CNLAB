import torch
from torch import nn
import os
import gluonnlp as nlp

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

# model_path = ".cache/et5/pytorch_model.bin"
# tokenizer_path = ".cache/et5/spice.model"
# t5model = T5ForConditionalGeneration.from_pretrained(model_path, return_dict = False)
# device = torch.device("cuda:0")
# t5model.to(device)
# t5model.eval()
# tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

model_config = T5Config.from_json_file(".cache/et5/config.json")
tokenizer_config = T5Config.from_json_file(".cache/et5/tokenizer_config.json")
model = T5ForConditionalGeneration(model_config)
state_dict = torch.load(".cache/et5/pytorch_model.bin")
model.load_state_dict(state_dict)
tokenizer = T5Tokenizer(vocab_file=".cache/et5/spiece.model", eos_token =  "</s>", unk_token = "<unk>", pad_token = "<pad>", extra_ids = 0)


file_num = 0

dataset_train = []
dataset_test = []

# Naming training_0, traing_1 ...
while(True):
    train_file = ".cache/training_"+ str(file_num) + ".txt"
    if not os.path.isfile(train_file):
        break

    dataset_train += [nlp.data.TSVDataset(train_file)]
    file_num += 1 
train_num = file_num

for i in range(train_num):
    for j in dataset_train[i][1:]:
        tmp = []
        encoded_dict = tokenizer(j[0])
        x = model(encoded_dict)
        print(x)
        print("---")