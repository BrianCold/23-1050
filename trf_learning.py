from utils import compute_metrics, extract_features, FeatureReducer
import hydra
import logging
import torch
import pandas as pd
from omegaconf import DictConfig
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_type = 'bert-tiny'

     # Load the fine-tuned sarcasm detection model and tokenizer
    sarcasm_model_directory = f"sd_models/{model_type}"
    sarcasm_model = BertForSequenceClassification.from_pretrained(sarcasm_model_directory).to(device)
    sarcasm_tokenizer = BertTokenizer.from_pretrained(sarcasm_model_directory)

    # Load the fine-tuned sentiment analysis model (Model B) and tokenizer
    sentiment_model_directory = f"sa_models/{model_type}"
    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_directory).to(device)
    sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_directory)

    # sentiment_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    # sentiment_model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=cfg.trf_lrg_params.n_classes, problem_type="multi_label_classification").to(device)

    def tokenize_function(examples):
        return sentiment_tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    sa_train_data = pd.read_csv(cfg.sa_data.train_data)
    sa_train_labels = pd.read_csv(cfg.sa_data.train_labels)
    sa_val_data = pd.read_csv(cfg.sa_data.val_data)
    sa_val_labels = pd.read_csv(cfg.sa_data.val_labels)
    sa_test_data = pd.read_csv(cfg.sa_data.test_data)
    sa_test_labels = pd.read_csv(cfg.sa_data.test_labels)

    # For train dataset
    sa_train = pd.concat([sa_train_data, sa_train_labels], axis=1)

    # For validation dataset
    sa_val = pd.concat([sa_val_data, sa_val_labels], axis=1)

    # For test dataset
    sa_test = pd.concat([sa_test_data, sa_test_labels], axis=1)

    sa_train.rename(columns={'clean_comment': 'text', 'category': 'labels'}, inplace=True)
    sa_val.rename(columns={'clean_comment': 'text', 'category': 'labels'}, inplace=True)
    sa_test.rename(columns={'clean_comment': 'text', 'category': 'labels'}, inplace=True)

    train_dataset = Dataset.from_pandas(sa_train)
    val_dataset = Dataset.from_pandas(sa_val)
    test_dataset = Dataset.from_pandas(sa_test)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Ensure tensors are properly formatted for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    dataset_dict = DatasetDict({
        'train': train_dataset.remove_columns(['text', 'token_type_ids']),
        'val': val_dataset.remove_columns(['text', 'token_type_ids']),
        'test': test_dataset.remove_columns(['text', 'token_type_ids'])
    })
    # print(dataset_dict)

    # Set both models to evaluation mode
    sarcasm_model.eval()
    sentiment_model.eval()

    # Prepare DataLoader
    train_loader = DataLoader(dataset_dict['train'], batch_size=cfg.trf_lrg_params.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_dict['val'], batch_size=cfg.trf_lrg_params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_dict['test'], batch_size=cfg.trf_lrg_params.batch_size, shuffle=False)

    # Define optimizer and criterion for the sentiment analysis model
    optimizer = torch.optim.AdamW(sentiment_model.parameters(), lr=cfg.trf_lrg_params.lr, weight_decay=cfg.trf_lrg_params.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # # Replace the classifier to accept 256 input features (128 from sarcasm + 128 from sentiment)
    # sentiment_model.classifier = nn.Linear(128 * 2, 3).to(device)

    # Instantiate the classifier
    fr = FeatureReducer(256, 128).to(device)

    for epochs in range(cfg.trf_lrg_params.epochs):
        sarcasm_model.eval()  # Ensure the sarcasm detection model is frozen
        sentiment_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        with tqdm(train_loader, desc=f"Epoch {epochs+1}", unit="batch") as tepoch:
            for batch in tepoch:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                labels = batch['labels'].to(device)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                # Extract features from both models
                sarcasm_features = extract_features(sarcasm_model, input_ids, attention_mask, token_type_ids)
                sentiment_features = extract_features(sentiment_model, input_ids, attention_mask, token_type_ids)

                # Average the features
                combined_features = (sarcasm_features + sentiment_features) / 2

                # # Concatenate the features 
                # combined_features = torch.cat((sarcasm_features, sentiment_features), dim=-1)

                # Extract the [CLS] token's hidden state from combined features (first token in each sequence)
                cls_combined_features = combined_features[:, 0, :]  # Shape: [batch_size, hidden_size]

                # pooled_features = fr(cls_combined_features)

                logits = sentiment_model.classifier(cls_combined_features)

                # Pooling after concatenation
                # # Max pooling over the sequence dimension 
                # pooled_features = torch.max(combined_features, dim=1).values  # Max pooling

                # # Alternatively, mean pooling over the sequence dimension
                # pooled_features = torch.mean(combined_features, dim=1)  # Mean pooling

                # pooled_features = fr(pooled_features)
                # # Pass the pooled features through the classifier
                # logits = sentiment_model.classifier(pooled_features)

                loss = criterion(logits, labels.long())
                train_loss = train_loss + loss

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                # print(predicted)
                # print(labels)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(train_loss=train_loss.item()/len(train_loader), train_accuracy=100*train_correct/train_total)
            
        sarcasm_model.eval()
        sentiment_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                labels = batch['labels'].to(device)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                # Extract features from both models
                sarcasm_features = extract_features(sarcasm_model, input_ids, attention_mask, token_type_ids)
                sentiment_features = extract_features(sentiment_model, input_ids, attention_mask, token_type_ids)

                # Average the features
                combined_features = (sarcasm_features + sentiment_features) / 2

                # # Concatenate the features 
                # combined_features = torch.cat((sarcasm_features, sentiment_features), dim=-1)

                # Extract the [CLS] token's hidden state from combined features (first token in each sequence)
                cls_combined_features = combined_features[:, 0, :]  # Shape: [batch_size, hidden_size]

                # pooled_features = fr(cls_combined_features)

                logits = sentiment_model.classifier(cls_combined_features)

                # Pooling after concatenation
                # # Max pooling over the sequence dimension 
                # pooled_features = torch.max(combined_features, dim=1).values  # Max pooling

                # # Alternatively, mean pooling over the sequence dimension
                # pooled_features = torch.mean(combined_features, dim=1)  # Mean pooling

                # pooled_features = fr(pooled_features)
                # # Pass the pooled features through the classifier
                # logits = sentiment_model.classifier(pooled_features)
                    
                # Calculate loss
                loss = criterion(logits, labels.long())
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        logger.info(
            "Epoch: {}, Validation loss: {:.5f}, Validation accuracy: {:.5f}".format(
            epochs + 1,
            val_loss / len(val_loader),
            val_correct / val_total *100
            )
        )

    sarcasm_model.eval()
    sentiment_model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            labels = batch['labels'].to(device)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            # Extract features from both models
            sarcasm_features = extract_features(sarcasm_model, input_ids, attention_mask, token_type_ids)
            sentiment_features = extract_features(sentiment_model, input_ids, attention_mask, token_type_ids)

            # Average the features
            combined_features = (sarcasm_features + sentiment_features) / 2

            # # Concatenate the features 
            # combined_features = torch.cat((sarcasm_features, sentiment_features), dim=-1)

            # Extract the [CLS] token's hidden state from combined features (first token in each sequence)
            cls_combined_features = combined_features[:, 0, :]  # Shape: [batch_size, hidden_size]

            # pooled_features = fr(cls_combined_features)

            logits = sentiment_model.classifier(cls_combined_features)

            # Pooling after concatenation
            # # Max pooling over the sequence dimension 
            # pooled_features = torch.max(combined_features, dim=1).values  # Max pooling

            # # Alternatively, mean pooling over the sequence dimension
            # pooled_features = torch.mean(combined_features, dim=1)  # Mean pooling

            # pooled_features = fr(pooled_features)
            # # Pass the pooled features through the classifier
            # logits = sentiment_model.classifier(pooled_features)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    logger.info(
        "Test accuracy: {:.5f}".format(
        test_correct / test_total * 100
        )
    )

if __name__ == "__main__":
    main()
