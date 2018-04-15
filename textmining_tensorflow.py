import tensorflow as tf
import numpy as np
import input 

X_train_tfidf, y_train, X_test_tfidf, y_test = input.preprocess_data()

targets_number = 20#len(newsgroups_train.target_names)
features_number = X_train_tfidf.shape[1]

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
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        #variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        #variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)

      print(activations.shape)
      print("ACTIVATIONSSSS")
      return activations

hidden1 = nn_layer(x_data, features_number, 500, 'layer1')

dropped = tf.nn.dropout(hidden1, 0.9)
model_output = nn_layer(dropped, 500, targets_number, 'layer2', act=tf.identity)

# Starting things adjacent to the model (loss, etc)

loss = tf.losses.sparse_softmax_cross_entropy(logits=model_output, labels=y_target)
tf.summary.scalar('loss', loss)

prediction = tf.argmax(model_output, 1)
predictions_correct = tf.equal(prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(predictions_correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

my_optimizer = tf.train.AdamOptimizer(0.001)
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
    rand_index = np.random.choice(X_train_tfidf.shape[0], size=128)
    rand_x = X_train_tfidf[rand_index].todense()#.toarray()#.todense()
    rand_y = np.transpose(y_train[rand_index])

    summary, _ = sess.run([merged, train_step], feed_dict={x_data: rand_x, y_target: rand_y})
    train_writer.add_summary(summary, i)

    if (i+1)%100==0:
        i_data.append(i+1)
        summary, train_loss_temp = sess.run([merged, loss], feed_dict={x_data: rand_x, y_target: rand_y})
        train_writer.add_summary(summary, i)
        train_loss.append(train_loss_temp)

        prediction_array = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y})
        print("Prediction: ")
        print(prediction_array)
        print("targets")
        print(rand_y)
        
        summary, train_acc_temp = sess.run([merged, accuracy], feed_dict={x_data: rand_x, y_target: rand_y})
        train_writer.add_summary(summary, i)
        train_acc.append(train_acc_temp)

        acc_and_loss = [i+1, train_loss_temp, train_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss: {:.2f}. Train Acc: {:.2f}'.format(*acc_and_loss))


