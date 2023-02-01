# KLUE-RoBERTa-base
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
