import tensorflow as tf
import numpy as np
import input 
import math
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import argparse

def preprocess_text(text_file):
    file_object = open(text_file, "r")
    text = file_object.read()
    # print(text)
    count_vect = joblib.load("tools/count_vect.pkl")
    X = count_vect.transform([text])
    tfidf_transformer = joblib.load("tools/tfidf_transformer.pkl")
    X = tfidf_transformer.transform(X)
    return X
        
def format_label(label):
    # print("Here we should format the label into something human readable")
    labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos',
                'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
                'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
                'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    
    return labels[label]

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="What model should I use for inference? Available options: random_forest/mlp")
parser.add_argument("--text_file", help="path the text file")
parser.add_argument("--scikit_model", help="path to saved scikit model (random forests)")
parser.add_argument("--tensorflow_model", help="path to saved tensorflow model (MLP)")
#parser.add_argument("--input_height", type=int, help="input height")
#parser.add_argument("--input_width", type=int, help="input width")
#parser.add_argument("--input_mean", type=int, help="input mean")
#parser.add_argument("--input_std", type=int, help="input std")
#parser.add_argument("--input_layer", help="name of input layer")
#parser.add_argument("--output_layer", help="name of output layer")
args = parser.parse_args()

random_forest_model_file = "random_forest_model.pkl"
tensorflow_model_directory = "./tensorflow_model/"
text_file = "test_textfile.txt"


if args.text_file:
    text_file = args.text_file
    
if args.model:
    model = args.model
    if model == "random_forest":
        if args.scikit_model:
            clf = joblib.load("random_forest_model.pkl")
            random_forest_model_file = args.scikit_model
            features = preprocess_text(text_file)
            label = clf.predict(features)
            print(format_label(label[0]))
    if model == "mlp":
        if args.tensorflow_model: 
            features = preprocess_text(text_file)
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(tensorflow_model_directory + 'mlp-1000.meta')
                saver.restore(sess, tf.train.latest_checkpoint(tensorflow_model_directory))
                graph = tf.get_default_graph()
                x_data = graph.get_tensor_by_name("X_data:0")
                prediction = graph.get_tensor_by_name("prediction:0")
                label = sess.run(prediction, feed_dict={x_data: features.todense()})
                #print(label[0])
                print(format_label(label[0]))