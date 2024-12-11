import hydra
from omegaconf import DictConfig
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
import torch
from utils import *

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    model_type = 'prajjwal1/bert-tiny'
    # model_type = 'distilbert-base-uncased'
    
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

    train_dataset = Dataset.from_pandas(sd_train)
    val_dataset = Dataset.from_pandas(sd_val)
    test_dataset = Dataset.from_pandas(sd_test)

    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertForSequenceClassification.from_pretrained(model_type, num_labels=cfg.sd_params.n_classes)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Ensure tensors are properly formatted for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })

    def save_model(model, path):
        # Make sure all parameters are contiguous
        for param in model.parameters():
            param.data = param.data.contiguous()

        # Save the model state dictionary
        torch.save(model.state_dict(), path)
    
    save_model(model, 'sd_model.pt')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=False,
        num_train_epochs=cfg.sd_params.epochs,
        per_device_train_batch_size=cfg.sd_params.batch_size,
        per_device_eval_batch_size=cfg.sd_params.batch_size,
        learning_rate = cfg.sd_params.lr,
        weight_decay=cfg.sd_params.weight_decay,
        logging_dir='./logs',
        logging_steps=1333,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on the validation dataset
    # validation_results = trainer.evaluate()

    # Temporarily set the test dataset as the eval_dataset
    trainer.eval_dataset = dataset_dict['test']

    # Evaluate on the test dataset
    test_results = trainer.evaluate()
    
    # Print the test results
    print(test_results)

    save = input('Save model? (y/n): ')

    if save == 'y':
        # Specify your custom save path
        save_directory = f'sd_models/{model_type}'

        # Save the trained model, tokenizer, and configuration to the specified directory
        trainer.save_model(save_directory)

        # Optionally, save the tokenizer separately if needed
        tokenizer.save_pretrained(save_directory)

        print('Model saved')


if __name__ == "__main__":
    main()