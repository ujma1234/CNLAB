import os
import time
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

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0")

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
        npzero = np.int32(0)
        label_list = [[npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero]]
        
        self.labels = [label_list] * data_num
        for i in range(data_num):
            label_list = [[npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero], [npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero,npzero]]
            month = int(dataset[i][0][0][0])*10 + int(dataset[i][0][0][1])
            day = int(dataset[i][0][0][3])*10 + int(dataset[i][0][0][4])
            hour = int(dataset[i][0][0][6])*10 + int(dataset[i][0][0][7])
            min = int(dataset[i][0][0][9])*10 + int(dataset[i][0][0][10])
            min = int(min/5)

            label_list[0][month-1] = np.int32(1)
            label_list[1][day-1] = np.int32(1)
            label_list[2][hour] = np.int32(1)
            label_list[3][min] = np.int32(1)

            self.labels[i] = label_list
            
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

############################################################################

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size * 64, 2048)
        self.fc = nn.Linear(2048, hidden_size)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1)
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.relu(out)
        out = self.fc(out)
        return out

class BERT(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERT, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return out

########################################################################

class LSBERT(nn.Module):
    ## hidden_size = 전달받는 은닉층의 크기, fc_size = 신경망 크기, num_layers = lstm_sell 크기
    def __init__(self, hidden_size, fc_size, num_layers):
        super(LSBERT, self).__init__()
        self.bert = BERT(bertmodel, dr_rate=0.5).to(device)
        self.f_lstm = LSTM(num_classes = 1, input_size = 768, hidden_size = hidden_size, num_layers = num_layers, seq_length = 768)
        self.num_classes = 4
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size

        self.month_classifier = nn.Linear(self.fc_size, 12)
        self.month_fc = nn.Linear(hidden_size, self.fc_size)
        self.day_classifier = nn.Linear(self.fc_size, 30)
        self.day_fc = nn.Linear(hidden_size, self.fc_size)
        self.hour_classifier = nn.Linear(self.fc_size, 24)
        self.hour_fc = nn.Linear(hidden_size, self.fc_size)
        self.min_classifier = nn.Linear(self.fc_size, 12)
        self.min_fc = nn.Linear(hidden_size, self.fc_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        PAD_pooler = torch.zeros(1, 768, dtype = torch.float32)
        pooler = []
        seq_len = 0
        for token_ids, valid_length, segment_ids in x:
            seq_len += 1
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            pooler += self.bert(token_ids, valid_length, segment_ids).tolist()

        for _ in range(64 - seq_len):
            pooler += PAD_pooler
        pooler = torch.tensor(pooler, dtype=torch.float32)
        pooler = pooler.reshape(1, 64, 768)

        out = self.f_lstm(pooler)
        print(out.size())

        m_out = self.month_fc(out)
        m_out = self.relu(m_out)
        m_out = self.month_classifier(m_out)
        d_out = self.day_fc(out)
        d_out = self.relu(d_out)
        d_out = self.day_classifier(d_out)
        h_out = self.hour_fc(out)
        h_out = self.relu(h_out)
        h_out = self.hour_classifier(h_out)
        mi_out = self.min_fc(out)
        mi_out = self.relu(mi_out)
        mi_out = self.min_classifier(mi_out)
        schedule_out = [m_out, d_out, h_out, mi_out]
        return schedule_out

##############################################################################

out3 = LSBERT(hidden_size = 768, num_layers=64, fc_size = 2048)

for batch_id, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    # pooler2 = torch.zeros(1,768, dtype = torch.float32)
    # pooler = []
    # seq_len = 0
    # for token_ids, valid_length, segment_ids in x:
    #     seq_len += 1
    #     token_ids = token_ids.long().to(device)
    #     segment_ids = segment_ids.long().to(device)
    #     valid_length= valid_length
    #     pooler += out(token_ids, valid_length, segment_ids).tolist()
    # for _ in range(64 - seq_len):
    #     pooler += pooler2
    # print(len(pooler))
    # pooler = torch.tensor(pooler, dtype=torch.float32)
    
    # pooler = pooler.reshape(1,64,768)
    # print(pooler)
    # output = out2.forward(pooler)

    output = out3(x)
    print(output)


num_epochs = 1000 
learning_rate = 0.0001 
input_size = 64
hidden_size = 768
num_layers = 1 
num_classes = 4

# model = LSBERT(num_classes, input_size, hidden_size, num_layers, 64) 

# criterion = torch.nn.MSELoss()    
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 



