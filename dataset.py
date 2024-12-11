import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gensim.downloader
from torch.nn.utils.rnn import pad_sequence
import subprocess

from sd import AutoTokenizer
class NLPDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, w2v) -> None:
        #Take in the data and labels from the csvs and convert them into a list
        data = pd.read_csv(x_path, usecols=[0]).squeeze().tolist()
        self.labels = pd.read_csv(y_path, usecols=[0]).squeeze().tolist()
        self.max_length = 0

        self.data = []
        for text in tqdm(data, desc="Converting text to index"):
            text_tokens = []
            for word in text.split(" "):
                try:
                    text_tokens.append(w2v.key_to_index[word])
                except KeyError:
                    pass
            self.data.append(torch.LongTensor(text_tokens))
        self.data = pad_sequence(self.data, batch_first=True, padding_value=0)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
# class NLPDataset(Dataset):
#     def __init__(self, x_path: str, y_path: str, tokenizer) -> None:
#         # Take in the data and labels from the csvs and convert them into a list
#         self.data = pd.read_csv(x_path, usecols=[0]).squeeze().tolist()
#         self.labels = pd.read_csv(y_path, usecols=[0]).squeeze().tolist()
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         text = self.data[index]
#         label = self.labels[index]
#         encoding = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        
#         # Flatten tensors to remove the extra dimension
#         item = {key: val.squeeze(0) for key, val in encoding.items()}
#         ordered_item = {
#             'labels': torch.tensor(label, dtype=torch.long),
#             'input_ids': item['input_ids'],
#             'token_type_ids': item['token_type_ids'],
#             'attention_mask': item['attention_mask']
#         }
#         # item['labels'] = 
        
        
#         # item = {key: torch.tensor(val[index]) for key, val in encoding.items()}
#         # # Adjust the order here
#         # ordered_item = {
#         #     'labels': torch.tensor(self.labels[index]),
#         #     'input_ids': item['input_ids'],
#         #     'token_type_ids': item['token_type_ids'],
#         #     'attention_mask': item['attention_mask']
#         # }
#         return ordered_item

if __name__ == "__main__":
    ds = NLPDataset(
        "sa_data/x_train.csv",
        "sa_data/y_train.csv",
        gensim.downloader.load("word2vec-google-news-300"),
    )
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
    # ds = NLPDataset(
    #     "sa_data/x_train.csv",
    #     "sa_data/y_train.csv",
    #     tokenizer,
    # )
    print(ds[0])