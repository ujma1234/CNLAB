import os
import gc
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import os.path
from tqdm.notebook import tqdm

from my_module import ps_bertNlstm

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0")

checkpoint = torch.load(".cache/test.zip")

bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
test_file = ".cache/test_"
file_num = 0

# Please Modify your data amount       --here--
dataset_train = [nlp.data.TSVDataset] *  100
dataset_test = [nlp.data.TSVDataset]  *  100

# Naming training_0, traing_1 ...
while(True):
    train_file = ".cache/training_"+ str(file_num) + ".txt"
    if not os.path.isfile(train_file):
        break

    dataset_train[file_num] = nlp.data.TSVDataset(train_file)
    file_num += 1 
train_num = file_num

file_num = 0

# Naming test_0, test_1 ...
while(True):
    test_file = ".cache/test_"+ str(file_num) + ".txt"
    if not os.path.isfile(test_file):
        break

    dataset_test[file_num] = nlp.data.TSVDataset(test_file)
    file_num += 1 
test_num = file_num

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset,data_num, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        # Make Embedding with Tokenizer
        self.sentences = [[transform([j][0]) for j in dataset[i][1:]] for i in range(data_num)]
        # Make labels 
        # npzero = np.int32(0)

        # label_list = [[npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero]]
        
        self.labels = []
        for i in range(data_num):
            # label_list = []
            # label_list += torch.zeros(1, 12, dtype = torch.int32).to(device)
            # label_list += torch.zeros(1, 31, dtype = torch.int32).to(device)
            # label_list += torch.zeros(1, 24, dtype = torch.int32).to(device)
            # label_list += torch.zeros(1, 12, dtype = torch.int32).to(device)

            # label_list = [[npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero]]
            month = int(dataset[i][0][0][0])*10 + int(dataset[i][0][0][1]) - 1
            day = int(dataset[i][0][0][3])*10 + int(dataset[i][0][0][4]) - 1
            hour = int(dataset[i][0][0][6])*10 + int(dataset[i][0][0][7])
            min = int(dataset[i][0][0][9])*10 + int(dataset[i][0][0][10])
            min = int(min/5)

            label_list = torch.tensor([month, day, hour, min]).to(device)
            # label_list[0][month-1] = np.int32(1)
            # label_list[1][day-1] = np.int32(1)
            # label_list[2][hour] = np.int32(1)
            # label_list[3][min] = np.int32(1)

            # print(label_list)

            self.labels += [label_list]
            
        # print(self.labels)

    def __getitem__(self, i):
        return ([self.sentences[i]] + [(self.labels[i])])

    def __len__(self):
        return (len(self.labels))

max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = BERTDataset(dataset_train, train_num, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, test_num, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)

bertmodel = bertmodel.to(device)

model = ps_bertNlstm.LSBERT(hidden_size = 768, fc_size = 2048, num_layers=64, bertmodel = bertmodel).to(device)

model.load_state_dict(checkpoint['model_state_dict'])

############################################################################################################################3

# def train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step=0):
#     for e in range(num_epochs):
#         train_acc = 0.0
#         test_acc = 0.0
#         losses = []
#         train_start_index = train_step + 1 if train_step != 0 else 0
#         total_train_step = len(train_loader)
#         model.train()
#         with tqdm(total = total_train_step, desc = f"Train({epoch})") as pbar:
#             pbar.update(train_step)
#             for batch_id, (x, label) in enumerate(train_loader, train_start_index):
#                 optimizer.zero_grad()
#                 outputs = model(x)

#                 loss

#########################################################################################################################

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def make_loss_N_Backward(data, label):
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    target = [label[0][0].reshape(1), label[0][1].reshape(1), label[0][2].reshape(1), label[0][3].reshape(1)]
    input = [data[0].reshape(1,12), data[1].reshape(1,31), data[2].reshape(1,24), data[3].reshape(1,12)]
    for i, j in zip(target, input) :
        loss = loss_fn(j, i)
        loss.backward(retain_graph=True)
        losses.append(loss.tolist())
    return losses

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

def calc_accuracy(X,Y):
    pred = []
    vals = []
    indices = []
    mon = X[0].reshape(1, 12)
    day = X[1].reshape(1, 31)
    hour = X[2].reshape(1, 24)
    min = X[3].reshape(1, 12)
    pred = [mon, day, hour, min]
    Y = Y[0]
    # print(Y)
    for i, j in (torch.max(k, 1) for k in pred):
        vals += i
        indices += j
        # print(j)
    indices = torch.tensor(indices).to(device)
    # print(indices.size()[0])
    train_acc = (indices == label).sum().data.cpu().numpy()/indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # label = torch.tensor(label)
        # labels = torch.tensor(label)
        # print(batch_id)
        predict = []
        out = model(x)
        # print(out, predict)
        loss = make_loss_N_Backward(out, label)
        # train_acc = calc_accuracy(out, label)
        # print(loss)
        # loss_mean = torch.mean(torch.stack(loss))
        # loss.backward()
        # # print(torch.mean(torch.stack(loss)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss, train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (x, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        label = label.long().to(device)
        out = model(x)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        
torch.save(checkpoint, '.cache/test.zip')