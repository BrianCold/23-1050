import gensim.downloader

def main():
    w2v = gensim.downloader.load("word2vec-google-news-300")
    pass

if __name__ == "__main__":
    main()
