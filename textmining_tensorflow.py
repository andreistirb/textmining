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
#dnn_feature_columns = []
#for key in range(0, X_train_array.shape[1]):
#    dnn_feature_columns.append("feature#" + str(key))

features_number = X_train_array.shape[1]

#X_train_dataframe = pd.DataFrame(
#    data=X_train_array,
#    columns=dnn_feature_columns
#)

#print(X_test_array.shape)

#X_test_dataframe = pd.DataFrame(
#    data=X_test_array,
#    columns=dnn_feature_columns
#)

sess = tf.Session()

# Starting logistic regression
A = tf.Variable(tf.random_normal(shape=[features_number, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[None, features_number], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

model_output = tf.add(tf.matmul(x_data, A), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

my_optimizer = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)

# Start logistic regression
train_loss = []
train_acc = []
i_data = []

train_writer = tf.summary.FileWriter('graph', sess.graph)
merged = tf.summary.merge_all()

for i in range(1000):
    rand_index = np.random.choice(X_train_tfidf.shape[0], size=64)
    rand_x = X_train_tfidf[rand_index].todense()
    rand_y = np.transpose([y_train[rand_index]])

    summary, _ = sess.run([merged, train_step], feed_dict={x_data: rand_x, y_target: rand_y})
    train_writer.add_summary(summary, i)

    if (i+1)%100==0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)
        
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        acc_and_loss = [i+1, train_loss_temp, train_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc: {:.2f}'.format(*acc_and_loss))


