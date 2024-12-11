from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiTaskBert(nn.Module):
    def __init__(self, model, sarcasm_num_labels, sentiment_num_labels) -> None:
        super().__init__()
        self.bert = model

        # Separate classification heads
        self.sarcasm_classifier = nn.Linear(self.bert.config.hidden_size, sarcasm_num_labels)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, sentiment_num_labels)
    
    def forward(self,input_ids, attention_mask, token_type_ids, task):
         # Forward pass through shared BERT backbone
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        # Extract [CLS] token representation
        cls_token_state = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    
        # Choose the correct classification head based on the task
        if task == 'sarcasm':
            logits = self.sarcasm_classifier(cls_token_state)
        elif task == 'sentiment':
            logits = self.sentiment_classifier(cls_token_state)
        else:
            raise ValueError("Task should be 'sarcasm' or 'sentiment'")
        
        return logits