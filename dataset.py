import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gensim.downloader

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
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

if __name__ == "__main__":
    ds = NLPDataset(
        "sd_data/x_train.csv",
        "sd_data/y_train.csv",
        gensim.downloader.load("word2vec-google-news-300"),
    )
    print(ds[0])


