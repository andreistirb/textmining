from pprint import pprint
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
#pprint(list(newsgroups_train.target_names))

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
#print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(newsgroups_train.data, newsgroups_train.target)

text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=50, random_state=42))])
_ = text_clf_svm.fit(newsgroups_train.data, newsgroups_train.target)

predicted = text_clf.predict(newsgroups_test.data)
predicted_svm = text_clf_svm.predict(newsgroups_test.data)
print("Naive Bayes")
print(np.mean(predicted == newsgroups_test.target))
print("Linear Classifier")
print(np.mean(predicted_svm == newsgroups_test.target))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(newsgroups_train.data, newsgroups_train.target)

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-2, 1e-3),}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(newsgroups_train.data, newsgroups_train.targets)

print("Best model for svm")
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)

print("Best model for linear")
print(gs_clf.best_score_)
print(gs_clf.best_params_)