# Format input for the models
# TO-DO: make this an object in order to preserve the countVectorizer for new text input by the user
# TO-DO: function to return batches from dataset in a more easy way

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

# Download data
def download_data():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    return newsgroups_train, newsgroups_test

# Preprocess data
def preprocess_data():
    newsgroups_train, newsgroups_test = download_data()

    # Instantiate the tools and preprocess train data
    try:
        count_vect = joblib.load("tools/count_vect.pkl")
        X_train_counts = count_vect.transform(newsgroups_train.data)
        # print("Count vector restored")
    except:
        count_vect = CountVectorizer(stop_words='english')
        X_train_counts = count_vect.fit_transform(newsgroups_train.data)
        joblib.dump(count_vect, "tools/count_vect.pkl")
    try:
        tfidf_transformer = joblib.load("tools/tfidf_transformer.pkl")
        X_train_tfidf = tfidf_transformer.transform(X_train_counts)
        #print("tfidf transformer restored")
    except:
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        joblib.dump(tfidf_transformer, "tools/tfidf_transformer.pkl")

    y_train = newsgroups_train.target

    # preprocess test data
    X_test_counts = count_vect.transform(newsgroups_test.data)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    y_test = newsgroups_test.target

    # print(X_train_tfidf.shape)
    return X_train_tfidf, y_train, X_test_tfidf, y_test
