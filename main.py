import pandas as pd
import json

sa_reddit_data = 'data/Reddit_Data.csv'
sa_twitter_data = 'data/Twitter_Data.csv'
df = pd.read_csv(sa_reddit_data)
print(len(df))
df = pd.read_csv(sa_twitter_data)
print(len(df))

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('data/Sarcasm_Headlines_Dataset.json'))
print(len(data))

