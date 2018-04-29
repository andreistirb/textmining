import tensorflow as tf
import numpy as np
import input 

X_train_tfidf, y_train, X_test_tfidf, y_test = input.preprocess_data()

targets_number = 20
features_number = X_train_tfidf.shape[1]
batch_size = 64

sess = tf.Session()
x_data = tf.placeholder(shape=[None, features_number], dtype=tf.float32, name="X_data")
y_target = tf.placeholder(tf.int64, [None], name="y_data")

# Starting the model
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

      return activations

hidden1 = nn_layer(x_data, features_number, 100, 'layer1')

dropped = tf.nn.dropout(hidden1, 0.9)
model_output = nn_layer(dropped, 100, targets_number, 'layer2', act=tf.identity)

# Starting things adjacent to the model (loss, etc)

loss = tf.losses.sparse_softmax_cross_entropy(logits=model_output, labels=y_target)
tf.summary.scalar('loss', loss)

prediction = tf.argmax(model_output, 1, name="prediction")
predictions_correct = tf.equal(prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(predictions_correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

my_optimizer = tf.train.AdamOptimizer(0.001)
train_step = my_optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init)

# Start model
train_loss = []
train_acc = []
i_data = []
test_acc = []

train_writer = tf.summary.FileWriter('graph', sess.graph)
merged = tf.summary.merge_all()
saver = tf.train.Saver()

for i in range(1000):
    rand_index = np.random.choice(X_train_tfidf.shape[0], size=batch_size)
    rand_x = X_train_tfidf[rand_index].todense()
    rand_y = np.transpose(y_train[rand_index])

    rand_index_test = np.random.choice(X_test_tfidf.shape[0], size=batch_size)
    rand_x_test = X_test_tfidf[rand_index_test].todense()
    rand_y_test = np.transpose(y_test[rand_index_test])

    summary, _ = sess.run([merged, train_step], feed_dict={x_data: rand_x, y_target: rand_y})
    train_writer.add_summary(summary, i)

    if (i+1)%100==0:
        i_data.append(i+1)
        summary, train_loss_temp = sess.run([merged, loss], feed_dict={x_data: rand_x, y_target: rand_y})
        train_writer.add_summary(summary, i)
        train_loss.append(train_loss_temp)

        # prediction_array = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y})
        
        summary, train_acc_temp = sess.run([merged, accuracy], feed_dict={x_data: rand_x, y_target: rand_y})
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x_test, y_target: rand_y_test})

        train_writer.add_summary(summary, i)
        train_acc.append(train_acc_temp)
        test_acc.append(test_acc_temp)

        acc_and_loss = [i+1, train_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc: {:.2f} Test Acc: {:.2f}'.format(*acc_and_loss))

# Test the model on the whole test set in order to compare with random forest
print("Testing the model...")
X_test = X_test_tfidf.todense()
y_test = np.transpose(y_test)
test_accuracy = sess.run(accuracy, feed_dict={x_data: X_test, y_target: y_test})
print("Final accuracy: %f"%test_accuracy)

#tf.saved_model.simple_save(sess, "tensorflow_model", inputs={"x": x_data}, outputs={"y": y_target} )
print("Saving the model")
saver.save(sess, './tensorflow_model/mlp', global_step=1000)

