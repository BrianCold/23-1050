import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description= "Creating train, validation and test datasets")
    parser.add_argument('--sa_dir', type = str, default = "sa_data/", help = 'Directory to save sentiment analysis dataset to')
    parser.add_argument('--sd_dir', type = str, default = "sd_data/", help = 'Directory to save sarcasm detection dataset to')
    parser.add_argument('--data_dir', type = str, default = "data/", help = 'Directory to load initial data from')

    return parser.parse_args()

def main(args: argparse.Namespace):
    
    #read the sentiment analysis datasets
    reddit_df = pd.read_csv(os.path.join(args.data_dir, "Reddit_Data.csv"))
    twitter_df = pd.read_csv(os.path.join(args.data_dir, "Twitter_Data.csv"))
    # Rename columns of twitter_df to match reddit_df
    twitter_df_renamed = twitter_df.rename(columns={'clean_text': 'clean_comment'})
    #combine the sentiment analysis datasets
    sa_df = pd.concat([reddit_df,twitter_df_renamed], axis=0, ignore_index=True)
    #split data into train, validation and test
    x_train, x_temp, y_train, y_temp = train_test_split(sa_df['clean_comment'], sa_df['category'],test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,test_size=0.5)
    #save split data into folder
    x_train.to_csv(os.path.join(args.sa_dir, 'x_train.csv'),index=False)
    x_val.to_csv(os.path.join(args.sa_dir, 'x_val.csv'),index=False)
    x_test.to_csv(os.path.join(args.sa_dir, 'x_test.csv'),index=False)
    y_train.to_csv(os.path.join(args.sa_dir, 'y_train.csv'),index=False)
    y_val.to_csv(os.path.join(args.sa_dir, 'y_val.csv'),index=False)
    y_test.to_csv(os.path.join(args.sa_dir, 'y_test.csv'),index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)