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

        # print(self.sentences)

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
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = BERTDataset(dataset_train, train_num, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, test_num, tok, max_len, True, False)


print(data_train)

############################################################################
# data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
# data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

# train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
# test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

# class BERTClassifier(nn.Module):
#     def __init__(self,
#                  bert,
#                  hidden_size = 768,
#                  num_classes=2,
#                  dr_rate=None,
#                  params=None):
#         super(BERTClassifier, self).__init__()
#         self.bert = bert
#         self.dr_rate = dr_rate
                 
#         self.classifier = nn.Linear(hidden_size , num_classes)
#         if dr_rate:
#             self.dropout = nn.Dropout(p=dr_rate)
    
#     def gen_attention_mask(self, token_ids, valid_length):
#         attention_mask = torch.zeros_like(token_ids)
#         for i, v in enumerate(valid_length):
#             attention_mask[i][:v] = 1
#         return attention_mask.float()

#     def forward(self, token_ids, valid_length, segment_ids):
#         attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
#         _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
#         if self.dr_rate:
#             out = self.dropout(pooler)
#         else:
#             out = pooler
#         return self.classifier(out)