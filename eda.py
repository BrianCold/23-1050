import matplotlib.pyplot as plt
import pandas as pd

def plot_sa_class_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame)-> None:

    fig, ax = plt.subplots(1,3)
    # Define the expected order of categories
    categories_order = sorted(df_train['category'].unique())

    # df_train['category'] = df_train['category'].astype(int)
    df_train['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0).plot(kind='barh', ax=ax[0])
    ax[0].set_title('Sentiment Analysis class distribution for Train labels')
    ax[0].set_xlabel('Count')
    ax[0].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_train['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0)):
        ax[0].text(value, index, str(value), va='center')

    # df_val['category'] = df_val['category'].astype(int)
    df_val['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0).plot(kind='barh', ax=ax[1])
    ax[1].set_title('Sentiment Analysis class distribution for Validation labels')
    ax[1].set_xlabel('Count')
    ax[1].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_val['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0)):
        ax[1].text(value, index, str(value), va='center')

    # df_test['category'] = df_test['category'].astype(int)
    df_test['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0).plot(kind='barh', ax=ax[2])
    ax[2].set_title('Sentiment Analysis class distribution for Test labels')
    ax[2].set_xlabel('Count')
    ax[2].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_test['category'].value_counts(ascending=True).reindex(categories_order, fill_value=0)):
        ax[2].text(value, index, str(value), va='center')

    plt.tight_layout()
    plt.show()

def plot_sd_class_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame)-> None:

    fig, ax = plt.subplots(1,3)

    df_train['is_sarcastic'] = df_train['is_sarcastic'].astype(int)
    df_train['is_sarcastic'].value_counts(ascending=True).plot(kind='barh', ax=ax[0])
    ax[0].set_title('Sarcasm Detection class distribution for Train labels')
    ax[0].set_xlabel('Count')
    ax[0].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_train['is_sarcastic'].value_counts(ascending=True)):
        ax[0].text(value, index, str(value), va='center')

    df_val['is_sarcastic'] = df_val['is_sarcastic'].astype(int)
    df_val['is_sarcastic'].value_counts(ascending=True).plot(kind='barh', ax=ax[1])
    ax[1].set_title('Sarcasm Detection class distribution for Validation labels')
    ax[1].set_xlabel('Count')
    ax[1].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_val['is_sarcastic'].value_counts(ascending=True)):
        ax[1].text(value, index, str(value), va='center')

    df_test['is_sarcastic'] = df_test['is_sarcastic'].astype(int)
    df_test['is_sarcastic'].value_counts(ascending=True).plot(kind='barh', ax=ax[2])
    ax[2].set_title('Sarcasm Detection class distribution for Test labels')
    ax[2].set_xlabel('Count')
    ax[2].set_ylabel('Category')
    # Annotating each bar with its count
    for index, value in enumerate(df_test['is_sarcastic'].value_counts(ascending=True)):
        ax[2].text(value, index, str(value), va='center')

    plt.tight_layout()
    plt.show()
    # print(df_test['is_sarcastic'].value_counts(ascending=True))

sa_train = pd.read_csv('sa_data/y_train.csv')
sa_val = pd.read_csv('sa_data/y_val.csv')
sa_test = pd.read_csv('sa_data/y_test.csv')

sd_train = pd.read_csv('sd_data/y_train.csv')
sd_val = pd.read_csv('sd_data/y_val.csv')
sd_test = pd.read_csv('sd_data/y_test.csv')

plot_sa_class_distribution(sa_train, sa_val, sa_test)
plot_sd_class_distribution(sd_train, sd_val, sd_test)