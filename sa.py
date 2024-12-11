import hydra
from omegaconf import DictConfig
import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import DatasetDict
from transformers import TrainingArguments
import torch
from utils import *
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(420)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_type = 'bert-tiny'
    
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

    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=cfg.sa_params.n_classes, problem_type="multi_label_classification").to(device)

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
    
    dataset_dict = DatasetDict({
        'train': train_dataset.remove_columns(['text', 'token_type_ids']),
        'val': val_dataset.remove_columns(['text', 'token_type_ids']),
        'test': test_dataset.remove_columns(['text', 'token_type_ids'])
    })

    def save_model(model, path):
        # Make sure all parameters are contiguous
        for param in model.parameters():
            param.data = param.data.contiguous()

        # Save the model state dictionary
        torch.save(model.state_dict(), path)
    
    save_model(model, 'sa_model.pt')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=False,
        num_train_epochs=cfg.sa_params.epochs,
        per_device_train_batch_size=cfg.sa_params.batch_size,
        per_device_eval_batch_size=cfg.sa_params.batch_size,
        learning_rate = cfg.sa_params.lr,
        weight_decay=cfg.sa_params.weight_decay,
        logging_dir='./logs',
        logging_steps=5000,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on the validation dataset
    validation_results = trainer.evaluate()

    # Temporarily set the test dataset as the eval_dataset
    trainer.eval_dataset = test_dataset

    # Evaluate on the test dataset
    test_results = trainer.evaluate()
    
    # Print the test results
    print(test_results)

    save = input('Save model? (y/n): ')

    if save == 'y':
        # Specify your custom save path
        save_directory = f'sa_models/{model_type}'

        # Save the trained model, tokenizer, and configuration to the specified directory
        trainer.save_model(save_directory)

        # Optionally, save the tokenizer separately if needed
        tokenizer.save_pretrained(save_directory)

        print('Model saved')

if __name__ == "__main__":
    main()