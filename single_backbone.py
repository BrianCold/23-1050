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
from model import MultiTaskBert
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_directory = 'prajjwal1/bert-tiny'
    model = BertForSequenceClassification.from_pretrained(model_directory, num_labels=cfg.trf_lrg_params.n_classes, problem_type="multi_label_classification")
    tokenizer = BertTokenizer.from_pretrained(model_directory)

    multitask_model = MultiTaskBert(model, sarcasm_num_labels=cfg.sd_params.n_classes, sentiment_num_labels=cfg.sa_params.n_classes).to(device)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

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

    sa_train_dataset = Dataset.from_pandas(sa_train)
    sa_val_dataset = Dataset.from_pandas(sa_val)
    sa_test_dataset = Dataset.from_pandas(sa_test)

    # Tokenize the datasets
    sa_train_dataset = sa_train_dataset.map(tokenize_function, batched=True)
    sa_val_dataset = sa_val_dataset.map(tokenize_function, batched=True)
    sa_test_dataset = sa_test_dataset.map(tokenize_function, batched=True)

    # Ensure tensors are properly formatted for PyTorch
    sa_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    sa_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    sa_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    sa_dataset_dict = DatasetDict({
        'train': sa_train_dataset.remove_columns(['text', 'token_type_ids']),
        'val': sa_val_dataset.remove_columns(['text', 'token_type_ids']),
        'test': sa_test_dataset.remove_columns(['text', 'token_type_ids'])
    })

    sd_train_data = pd.read_csv(cfg.sd_data.train_data)
    sd_train_labels = pd.read_csv(cfg.sd_data.train_labels)
    sd_val_data = pd.read_csv(cfg.sd_data.val_data)
    sd_val_labels = pd.read_csv(cfg.sd_data.val_labels)
    sd_test_data = pd.read_csv(cfg.sd_data.test_data)
    sd_test_labels = pd.read_csv(cfg.sd_data.test_labels)

    # For train dataset
    sd_train = pd.concat([sd_train_data, sd_train_labels], axis=1)

    # For validation dataset
    sd_val = pd.concat([sd_val_data, sd_val_labels], axis=1)

    # For test dataset
    sd_test = pd.concat([sd_test_data, sd_test_labels], axis=1)

    sd_train.rename(columns={'headline': 'text', 'is_sarcastic': 'labels'}, inplace=True)
    sd_val.rename(columns={'headline': 'text', 'is_sarcastic': 'labels'}, inplace=True)
    sd_test.rename(columns={'headline': 'text', 'is_sarcastic': 'labels'}, inplace=True)

    sd_train_dataset = Dataset.from_pandas(sd_train)
    sd_val_dataset = Dataset.from_pandas(sd_val)
    sd_test_dataset = Dataset.from_pandas(sd_test)

    # print(sd_train_dataset)
    # Tokenize the datasets
    sd_train_dataset = sd_train_dataset.map(tokenize_function, batched=True)
    sd_val_dataset = sd_val_dataset.map(tokenize_function, batched=True)
    sd_test_dataset = sd_test_dataset.map(tokenize_function, batched=True)

    # Ensure tensors are properly formatted for PyTorch
    sd_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    sd_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    sd_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    sd_dataset_dict = DatasetDict({
        'train': sd_train_dataset.remove_columns(['text', 'token_type_ids']),
        'val': sd_val_dataset.remove_columns(['text', 'token_type_ids']),
        'test': sd_test_dataset.remove_columns(['text', 'token_type_ids'])
    })
    # print(dataset_dict)

    # Prepare DataLoader
    sa_train_loader = DataLoader(sa_dataset_dict['train'], batch_size=cfg.sa_params.batch_size, shuffle=True)
    sa_val_loader = DataLoader(sa_dataset_dict['val'], batch_size=cfg.sa_params.batch_size, shuffle=True)
    sa_test_loader = DataLoader(sa_dataset_dict['test'], batch_size=cfg.sa_params.batch_size, shuffle=False)

    # Prepare DataLoader
    sd_train_loader = DataLoader(sd_dataset_dict['train'], batch_size=cfg.sd_params.batch_size, shuffle=True)
    sd_val_loader = DataLoader(sd_dataset_dict['val'], batch_size=cfg.sd_params.batch_size, shuffle=True)
    sd_test_loader = DataLoader(sd_dataset_dict['test'], batch_size=cfg.sd_params.batch_size, shuffle=False)

    # Define optimizer and criterion for the sentiment analysis model
    optimizer = torch.optim.AdamW(multitask_model.parameters(), lr=cfg.trf_lrg_params.lr, weight_decay=cfg.trf_lrg_params.weight_decay)
    
    sa_loss_fn = torch.nn.CrossEntropyLoss()
    sd_loss_fn = torch.nn.CrossEntropyLoss()

    # Step 2: Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    a = cfg.trf_lrg_params.a
    b = cfg.trf_lrg_params.b

    for epochs in range(cfg.trf_lrg_params.epochs):
        multitask_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        sa_loss_total = 0
        sa_correct_total = 0
        sa_total = 0

        sd_loss_total = 0
        sd_correct_total = 0
        sd_total = 0

        with tqdm(zip(sa_train_loader, sd_train_loader), desc=f"Epoch {epochs+1}", unit="batch") as tepoch:
            for sa_batch, sd_batch in tepoch:
                sa_input_ids = sa_batch['input_ids'].to(device)
                sa_attention_mask = sa_batch['attention_mask'].to(device)
                sa_token_type_ids = sa_batch.get('token_type_ids', None)
                sa_labels = sa_batch['labels'].to(device)

                if sa_token_type_ids is not None:
                    sa_token_type_ids = sa_token_type_ids.to(device)
                
                sa_logits = multitask_model(input_ids=sa_input_ids, attention_mask=sa_attention_mask, token_type_ids=sa_token_type_ids, task='sentiment')
                
                sa_loss = sa_loss_fn(sa_logits, sa_labels.long())
                sa_correct = (sa_logits.argmax(dim=-1) == sa_labels.long()).sum().item()

                sd_input_ids = sd_batch['input_ids'].to(device)
                sd_attention_mask = sd_batch['attention_mask'].to(device)
                sd_token_type_ids = sd_batch.get('token_type_ids', None)
                sd_labels = sd_batch['labels'].to(device)

                if sd_token_type_ids is not None:
                    sd_token_type_ids = sd_token_type_ids.to(device)
                
                sd_logits = multitask_model(input_ids=sd_input_ids, attention_mask=sd_attention_mask, token_type_ids=sd_token_type_ids, task='sarcasm')
                
                sd_loss = sd_loss_fn(sd_logits, sd_labels.long())
                sd_correct = (sd_logits.argmax(dim=-1) == sd_labels.long()).sum().item()

                sa_loss_total += sa_loss.item()
                sa_correct_total += sa_correct
                sa_total += sa_input_ids.size(0)

                sd_loss_total += sd_loss.item()
                sd_correct_total += sd_correct
                sd_total += sd_input_ids.size(0)

                optimizer.zero_grad()
                ((a*sa_loss) + (b*sd_loss)).backward()
                optimizer.step()

                # Update tqdm progress bar with loss and accuracy
                tepoch.set_postfix(Sarcasm_train_accuracy=(sd_correct_total / sd_total * 100), Sarcasm_train_loss=(sd_loss_total / sd_total), Sentiment_train_loss=(sa_loss_total / sa_total), Sentiment_train_accuracy=(sa_correct_total / sa_total * 100))
        
        multitask_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        sa_loss_total = 0
        sa_correct_total = 0
        sa_total = 0

        sd_loss_total = 0
        sd_correct_total = 0
        sd_total = 0

        with torch.no_grad():
            for (sa_batch, sd_batch) in zip(sa_val_loader, sd_val_loader):
                sa_input_ids = sa_batch['input_ids'].to(device)
                sa_attention_mask = sa_batch['attention_mask'].to(device)
                sa_token_type_ids = sa_batch.get('token_type_ids', None)
                sa_labels = sa_batch['labels'].to(device)

                if sa_token_type_ids is not None:
                    sa_token_type_ids = sa_token_type_ids.to(device)
                
                sa_logits = multitask_model(input_ids=sa_input_ids, attention_mask=sa_attention_mask, token_type_ids=sa_token_type_ids, task='sentiment')
                
                sa_loss = sa_loss_fn(sa_logits, sa_labels.long())
                sa_correct = (sa_logits.argmax(dim=-1) == sa_labels.long()).sum().item()

                sd_input_ids = sd_batch['input_ids'].to(device)
                sd_attention_mask = sd_batch['attention_mask'].to(device)
                sd_token_type_ids = sd_batch.get('token_type_ids', None)
                sd_labels = sd_batch['labels'].to(device)

                if sd_token_type_ids is not None:
                    sd_token_type_ids = sd_token_type_ids.to(device)
                
                sd_logits = multitask_model(input_ids=sd_input_ids, attention_mask=sd_attention_mask, token_type_ids=sd_token_type_ids, task='sarcasm')
                sd_loss = sd_loss_fn(sd_logits, sd_labels.long())
                sd_correct = (sd_logits.argmax(dim=-1) == sd_labels.long()).sum().item()

                # Backpropagation
                val_loss += (sa_loss.item() + sd_loss.item())
                val_correct += (sa_correct + sd_correct)
                val_total += sa_input_ids.size(0) + sd_input_ids.size(0)

                sa_loss_total += sa_loss.item()
                sa_correct_total += sa_correct
                sa_total += sa_input_ids.size(0)

                sd_loss_total += sd_loss.item()
                sd_correct_total += sd_correct
                sd_total += sd_input_ids.size(0)

        scheduler.step(val_loss/val_total)  # Reduce learning rate if val_loss plateaus

        logger.info(
            "Epoch: {}, Sentiment validation loss: {:.5f}, Sentiment validation accuracy: {:.5f}, Sarcasm validation loss: {:.5f}, Sarcasm validation accuracy: {:.5f}".format(
            epochs + 1,
            sa_loss_total / sa_total,
            sa_correct_total / sa_total *100,
            sd_loss_total / sd_total,
            sd_correct_total / sd_total *100
            )
        )
    
    multitask_model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    sa_loss_total = 0
    sa_correct_total = 0
    sa_total = 0

    sd_loss_total = 0
    sd_correct_total = 0
    sd_total = 0

    with torch.no_grad():
        for (sa_batch, sd_batch) in zip(sa_test_loader, sd_test_loader):
            sa_input_ids = sa_batch['input_ids'].to(device)
            sa_attention_mask = sa_batch['attention_mask'].to(device)
            sa_token_type_ids = sa_batch.get('token_type_ids', None)
            sa_labels = sa_batch['labels'].to(device)

            if sa_token_type_ids is not None:
                sa_token_type_ids = sa_token_type_ids.to(device)
            
            sa_logits = multitask_model(input_ids=sa_input_ids, attention_mask=sa_attention_mask, token_type_ids=sa_token_type_ids, task='sentiment')
            sa_loss = sa_loss_fn(sa_logits, sa_labels.long())
            sa_correct = (sa_logits.argmax(dim=-1) == sa_labels.long()).sum().item()

            # Backpropagation
            test_loss += sa_loss.item()
            test_correct += sa_correct
            test_total += sa_input_ids.size(0)

    logger.info(
        "Test loss: {:.5f}, Test accuracy: {:.5f}".format(
        test_loss / test_total,
        test_correct / test_total *100
        )
    )

if __name__ == "__main__":
    main()
