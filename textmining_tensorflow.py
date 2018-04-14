import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Download data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# Preprocess train data
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(newsgroups_train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_array = X_train_tfidf.toarray()
y_train = newsgroups_train.target

# Preprocess test data
#X_test_counts = count_vect.fit_transform(newsgroups_test.data)
#X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

#X_test_array = X_test_tfidf.toarray()
#y_test = newsgroups_test.target

# Make dataframes from input
dnn_feature_columns = []
for key in range(0, X_train_array.shape[1]):
    dnn_feature_columns.append("feature#" + str(key))

X_train_dataframe = pd.DataFrame(
    data=X_train_array,
    columns=dnn_feature_columns
)

#print(X_test_array.shape)

#X_test_dataframe = pd.DataFrame(
#    data=X_test_array,
#    columns=dnn_feature_columns
#)

# print(df.head())

# Feed the data to the model

#### DNN Clasifier

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

dnn_feature_columns_tf = []
for key in range(0, X_train_array.shape[1]):
    dnn_feature_columns_tf.append(tf.feature_column.numeric_column(key=key))

dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns=dnn_feature_columns_tf,
    hidden_units=[10, 10],
    n_classes=len(newsgroups_train.target_names)
)

dnn_classifier.train(
    input_fn=lambda:train_input_fn(X_train_dataframe, y_train, 16),
    steps=1000
)

#eval_result = dnn_classifier.evaluate(
#    input_fn=lambda:eval_input_fn(X_test_dataframe, y_test, 16)
#)

#print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
