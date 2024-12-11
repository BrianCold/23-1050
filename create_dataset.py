import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import numpy as np
import json
import csv
import re
import contractions
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Define the text cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description= "Creating train, validation and test datasets")
    parser.add_argument('--sa_dir', type = str, default = "sa_data/", help = 'Directory to save sentiment analysis dataset to')
    parser.add_argument('--sd_dir', type = str, default = "sd_data/", help = 'Directory to save sarcasm detection dataset to')
    parser.add_argument('--data_dir', type = str, default = "data/", help = 'Directory to load initial data from')

    return parser.parse_args()

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

def save_to_csv(data, filename, header=None):
    """Function to save data to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def clean_sd_data(text):

    # Expand contractions
    text = contractions.fix(text)
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    return text

def clean_sa_data(text):
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

def main(args: argparse.Namespace):
    
    # #read the sentiment analysis datasets
    # reddit_df = pd.read_csv(os.path.join(args.data_dir, "Reddit_Data.csv"))
    # twitter_df = pd.read_csv(os.path.join(args.data_dir, "Twitter_Data.csv"))
    # # Rename columns of twitter_df to match reddit_df
    # twitter_df = twitter_df.rename(columns={'clean_text': 'clean_comment'})
    # #convert data to string and replace nan strings with empty strings (eda)
    # reddit_df['clean_comment'] = reddit_df['clean_comment'].astype(str).replace('nan', '')
    # twitter_df['clean_comment'] = twitter_df['clean_comment'].astype(str).replace('nan', '')
    # # print(reddit_df.head())
    # # print(reddit_df.isnull().sum())
    # # print(twitter_df.head())
    # # print(twitter_df.isnull().sum())
    # #clean input data
    # reddit_df['clean_comment'] = reddit_df['clean_comment'].apply(clean_sa_data)
    # twitter_df['clean_comment'] = twitter_df['clean_comment'].apply(clean_sa_data)
    # #remove empty data
    # reddit_df = reddit_df[~reddit_df['clean_comment'].isnull() & (reddit_df['clean_comment'] != '')].reset_index(drop=True)
    # twitter_df = twitter_df[~twitter_df['clean_comment'].isnull() & (twitter_df['clean_comment'] != '')].reset_index(drop=True)
    # reddit_df = reddit_df[~reddit_df['category'].isnull() & (reddit_df['category'] != '')].reset_index(drop=True)
    # twitter_df = twitter_df[~twitter_df['category'].isnull() & (twitter_df['category'] != '')].reset_index(drop=True)
    # #combine the sentiment analysis datasets
    # # sa_df = pd.concat([reddit_df,twitter_df], axis=0, ignore_index=True)
    # sa_df = twitter_df
    # sa_df['category'] = sa_df['category'].replace(-1.0, 2.0)
    # #split data into train, validation and test
    # x_train, x_temp, y_train, y_temp = train_test_split(sa_df['clean_comment'], sa_df['category'],test_size=0.2, random_state=42)
    # x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,test_size=0.5, random_state=42)
    # # #clean input data
    # # x_train = x_train.apply(clean_text)
    # # x_val = x_val.apply(clean_text)
    # # x_test = x_test.apply(clean_text)
    # #save split data into folder
    # x_train.to_csv(os.path.join(args.sa_dir, 'x_train.csv'),index=False)
    # x_val.to_csv(os.path.join(args.sa_dir, 'x_val.csv'),index=False)
    # x_test.to_csv(os.path.join(args.sa_dir, 'x_test.csv'),index=False)
    # y_train.to_csv(os.path.join(args.sa_dir, 'y_train.csv'),index=False)
    # y_val.to_csv(os.path.join(args.sa_dir, 'y_val.csv'),index=False)
    # y_test.to_csv(os.path.join(args.sa_dir, 'y_test.csv'),index=False)

    headlines = []
    labels = []

    for entry in parse_data('data/Sarcasm_Headlines_Dataset_v2.json'):
        headlines.append(entry['headline'])
        labels.append(entry['is_sarcastic'])
    x_train, x_temp, y_train, y_temp = train_test_split(headlines, labels, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,test_size=0.5, random_state=42)
    # Convert the split data into DataFrames
    train_df = pd.DataFrame({'headline': x_train, 'is_sarcastic': y_train})
    val_df = pd.DataFrame({'headline': x_val, 'is_sarcastic': y_val})
    test_df = pd.DataFrame({'headline': x_test, 'is_sarcastic': y_test})
    #clean input data
    train_df['headline'] = train_df['headline'].apply(clean_sd_data)
    val_df['headline'] = val_df['headline'].apply(clean_sd_data)
    test_df['headline'] = test_df['headline'].apply(clean_sd_data)
    #remove empty data
    train_df = train_df[train_df['headline'].str.strip().astype(bool)]
    val_df = val_df[val_df['headline'].str.strip().astype(bool)]
    test_df = test_df[test_df['headline'].str.strip().astype(bool)]
    # Save the DataFrames to CSV files
    train_df[['headline']].to_csv(os.path.join(args.sd_dir, 'x_train.csv'), index=False)
    train_df[['is_sarcastic']].to_csv(os.path.join(args.sd_dir, 'y_train.csv'), index=False)
    val_df[['headline']].to_csv(os.path.join(args.sd_dir, 'x_val.csv'), index=False)
    val_df[['is_sarcastic']].to_csv(os.path.join(args.sd_dir, 'y_val.csv'), index=False)
    test_df[['headline']].to_csv(os.path.join(args.sd_dir, 'x_test.csv'), index=False)
    test_df[['is_sarcastic']].to_csv(os.path.join(args.sd_dir, 'y_test.csv'), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)