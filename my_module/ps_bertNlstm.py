import torch
from torch import nn

from my_module import ps_lstm
from my_module import ps_bert

device = torch.device("cuda:0")

class LSBERT(nn.Module):
    ## hidden_size = 전달받는 은닉층의 크기, fc_size = 신경망 크기, num_layers = lstm_sell 크기
    def __init__(self, hidden_size, fc_size, num_layers, bertmodel):
        super(LSBERT, self).__init__()
        self.bert = ps_bert.BERT(bertmodel, dr_rate=0.5).to(device)
        self.f_lstm = ps_lstm.LSTM(num_classes = 1, input_size = 768, hidden_size = hidden_size, num_layers = num_layers, seq_length = 768)
        self.num_classes = 4
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size

        self.month_classifier = nn.Linear(self.fc_size, 12)
        self.month_fc = nn.Linear(hidden_size, self.fc_size)
        self.day_classifier = nn.Linear(self.fc_size, 31)
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

        out = self.f_lstm(pooler, seq_len)
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