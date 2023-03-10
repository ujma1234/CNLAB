# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from torch.autograd import Variable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
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

# Please Modify your data amount        --here--
dataset_train = [nlp.data.TSVDataset] *   100
dataset_test = [nlp.data.TSVDataset]  *   100

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

# Construct Dataset
class BERTDataset(Dataset):
    def __init__(self, dataset,data_num, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        # Make Embedding with Tokenizer
        self.sentences = [[transform([j][0]) for j in dataset[i][1:]] for i in range(data_num)]
        
        # Make labels 
        self.labels = []
        for i in range(data_num):
            month = int(dataset[i][0][0][0])*10 + int(dataset[i][0][0][1]) - 1
            day = int(dataset[i][0][0][3])*10 + int(dataset[i][0][0][4]) - 1
            hour = int(dataset[i][0][0][6])*10 + int(dataset[i][0][0][7])
            min = int(dataset[i][0][0][9])*10 + int(dataset[i][0][0][10])
            min = int(min/5)

            label_list = torch.tensor([month, day, hour, min]).to(device)

            self.labels += [label_list]

    def __getitem__(self, i):
        return ([self.sentences[i]] + [(self.labels[i])])

    def __len__(self):
        return (len(self.labels))

# Setting Hyper Parameter
dr_rates = [0.3, 0.3, 0.3]
max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# data Load
data_train = BERTDataset(dataset_train, train_num, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, test_num, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)

# Model Load
bertmodel = bertmodel.to(device)
model = ps_bertNlstm.LSBERT(hidden_size = 768, fc_size = 2048, num_layers=1, bertmodel = bertmodel, dr_rate = dr_rates).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# Get Loss and Backward each output
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

# Setting checkpoint
checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

# Calculate accuracy
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
    for i, j in (torch.max(k, 1) for k in pred):
        vals += i
        indices += j
    indices = torch.tensor(indices).to(device)
    train_acc = (indices == label).sum().data.cpu().numpy()/indices.size()[0]
    return train_acc

# Training and Evaluate
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        predict = []
        out = model(x)
        loss = make_loss_N_Backward(out, label)
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
        
# Save checkpoint at the end
torch.save(checkpoint, '.cache/test.zip')
