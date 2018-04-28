import tensorflow as tf
import numpy as np
import input 
import math
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

X_train_tfidf, y_train, X_test_tfidf, y_test = input.preprocess_data()
targets_number = 20
features_number = X_train_tfidf.shape[1]
batch_size = 64

try:
    clf = joblib.load("random_forest_model.pkl")
    print("Yahoo, model restored")
except:
    print("Error no file found, starting from beginning")
    
    clf = RandomForestClassifier(n_estimators=5000, max_features=int(math.sqrt(features_number)))
    clf = clf.fit(X_train_tfidf, y_train)
    joblib.dump(clf, "random_forest_model.pkl")
    
score = clf.score(X_test_tfidf, y_test)
print("Score of RandomForest is: %f" % score)



