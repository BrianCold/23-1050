from sklearn.metrics import accuracy_score
import torch
from transformers import Trainer
from torch import nn

# Define Custom Compute Metrics Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def extract_features(model, input_ids, attention_mask, token_type_ids=None):
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        my_custom_loss = nn.CrossEntropyLoss()
        if return_outputs:
            return my_custom_loss(logits, labels.long()), outputs
        else:
            return my_custom_loss(logits, labels.long())

def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
        }

class FeatureReducer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureReducer, self).__init__()
        # Linear layer to map hidden states to number of labels
        self.classifier = nn.Linear(input_size, hidden_size)
    
    def forward(self, pooled_output):
        # Apply the linear layer to the pooled output (e.g., [CLS] token's hidden state)
        logits = self.classifier(pooled_output)
        return logits