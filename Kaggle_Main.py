
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from logisticClassify2 import *
from sklearn import ensemble as sk_ensemble
from sklearn import tree
from sklearn import neural_network
from sklearn import svm


np.random.seed(0)

X_test = np.genfromtxt("data/X_test.txt",delimiter=None)
X = np.genfromtxt("data/X_train.txt",delimiter=None)
Y = np.genfromtxt("data/Y_train.txt",delimiter=None)

X,Y = ml.shuffleData(X,Y)
# X = X[:20000]
# Y = Y[:20000]
[Xtr,Xva,Ytr,Yva] = ml.splitData(X,Y,0.75)
train_shape = Xtr.shape[0]

learner_list = []

x,y = ml.bootstrapData(Xtr, Ytr, train_shape)
x_val,y_val = ml.bootstrapData(Xva, Yva, train_shape)


# In[8]:

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import itertools

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

def preprocess(X_tr, X_va, n_features=None, weights=None, normalize=True):
    if not weights:
        model = ExtraTreesClassifier()
        model.fit(X_tr, Y_tr)
        percentages = model.feature_importances_
        max_p = max(percentages)
        weights = np.array([p/max_p for p in percentages])
    if normalize:
        X_tr = (X_tr - X_tr.min(axis=0))/(X_tr.max(axis=0))
        X_va = (X_va - X_va.min(axis=0))/(X_va.max(axis=0))
    X_tr *= weights
    X_va *= weights
    if n_features:
        pca = PCA(n_components=n_features)
        pca.fit(X_tr)
        X_tr = pca.transform(X_tr)
        pca.fit(X_va)
        X_va = pca.transform(X_va)
    return X_tr, X_va

X_te = np.genfromtxt("data/X_test.txt",delimiter=None)
X_tr = np.genfromtxt("data/X_train.txt",delimiter=None)
Y_tr = np.genfromtxt("data/Y_train.txt",delimiter=None)

X_tr, X_va, Y_tr, Y_va = ml.splitData(X_tr, Y_tr, 0.75)


# print("Original X_tr")
# print("---------------------------------------------------------------------------")
# print("Number of Observations: {}".format(X_tr.shape[0]))
# print("Number of features: {}\n".format(X_tr.shape[1]))
# print(X_tr)
# print()

#no preprocessing at all gets the highest auc score for k = 15

# #no preprocessing at all
# #k_list = [10, 15, 55, 100, 500, 1000, 3000]
# k_list = [15]
# for k in k_list:
    
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_tr, Y_tr)
YvaHat = knn.predict(X_va)
print("AUC for KNN is {}".format(roc_auc_score(Y_va, YvaHat)))
# print("Accuracy Score for Adaboost is {}".format(accuracy_score(Y_va, YvaHat)))
# knn.fit(X_va, Y_va)
# YvaHat = knn.predict(X_tr)
# print()
YvaHat = knn.predict(X_test)
learner_list.append(YvaHat)


# #with full on preprocessing
# k_list = [400, 500]
# for n in [2, 3, 4]:
#     print("number of features: {}".format(n))
#     X_tr_prime, X_va_prime = preprocess(X_tr, X_va, n_features=n)
#     X_tr_prime, X_te_prime = preprocess(X_tr, X_te, n_features=n)
#     print("X_tr post-preprocessing")
#     print(X_tr_prime)
#     print()
#     for k in k_list:
#         knn = KNeighborsRegressor(n_neighbors=k)
#         knn.fit(X_tr_prime, Y_tr)
#         YvaHat = knn.predict(X_va_prime)
#         print("For k={}, AUC is {}".format(k, roc_auc_score(Y_va, YvaHat)))
#         YvaHat = knn.predict(X_te_prime)
#         learner_list.append(YvaHat)
#     print()
    


# In[3]:

#Adabooster:
from sklearn.ensemble import AdaBoostRegressor
ada_clf = AdaBoostRegressor(base_estimator = tree.ExtraTreeClassifier(),n_estimators = 100, learning_rate = 0.5)
ada_clf.fit(X_tr, Y_tr)
# ada_predictions = ada_clf.predict(X_va)
# print("AUC for Adaboost is {}".format(roc_auc_score(Y_va, ada_predictions)))
# print("MSE_validation for Adaboost is {}".format(mean_squared_error(Y_va, ada_predictions)))
# ada_clf.fit(X_va, Y_va)
# ada_predictions = ada_clf.predict(X_tr)
# print("MSE_train for Adaboost is {}".format(mean_squared_error(Y_tr, ada_predictions)))
# print()
ada_predictions = ada_clf.predict(X_test)
learner_list.append(ada_predictions)
'''
Previous scores:
AdaBoostClassifier() = 0.680875
AdaBoostClassifier(n_estimators = 100) = 0.6795625
AdaBoostClassifier(n_estimators = 100, learning_rate = 10) = 0.344375
AdaBoostClassifier(n_estimators = 100, learning_rate = 0.5) = 0.681
AdaBoostClassifier(n_estimators = 100, learning_rate = 0.1) = 0.6771875
AdaBoostClassifier(n_estimators = 25, learning_rate = 0.5) = 0.675875
AdaBoostClassifier(base_estimator = tree.ExtraTreeClassifier(),n_estimators = 100, learning_rate = 0.5) = 0.6735625
AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(),n_estimators = 100, learning_rate = 0.5) = 0.681375
'''


# In[4]:


'''
Linear regression
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
Don't know if any other model would work
'''

#Random Forest Classifier:
x,y = ml.bootstrapData(Xtr, Ytr, train_shape)
x_val,y_val = ml.bootstrapData(Xva, Yva, train_shape)
dtree = sk_ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 50, min_samples_leaf = 2**3)
dtree.fit(X_tr, Y_tr)
# tree_predict = dtree.predict(X_va)
# print("AUC for tree_predict is {}".format(roc_auc_score(Y_va, tree_predict)))
# print("MSE_validation for Random Forest is {}".format(mean_squared_error(Y_va, tree_predict)))
# dtree.fit(X_va, Y_va)
# tree_predict = dtree.predict(X_tr)
# print("MSE_train for Random Forest is {}".format(mean_squared_error(Y_tr, tree_predict)))
# print()
tree_predict = dtree.predict(X_test)
learner_list.append(tree_predict)
'''
List of Previous Scores:
RandomForestClassifier(n_estimators = 50, max_depth = 15, min_samples_leaf = 2**7) == 0.6873125
RandomForestClassifier(n_estimators = 50, max_depth = 30, min_samples_leaf = 2**7) == 0.692875
RandomForestClassifier(n_estimators = 50, max_depth = 50, min_samples_leaf = 2**7) == 0.692875
RandomForestClassifier(n_estimators = 50, max_depth = 30, min_samples_leaf = 2**10) == 0.6818125
RandomForestClassifier(n_estimators = 100, max_depth = 30, min_samples_leaf = 2**7) == 0.6890625
.RandomForestClassifier(n_estimators = 10, max_depth = 30, min_samples_leaf = 2**7) == 0.6868125
'''


# In[5]:

#Final Blending Method:
Ypred = []
review_list = learner_list
for each_list in review_list:
    print(len(each_list))
new_learner_list = np.array(learner_list)
print(new_learner_list)
for each_item in range(len(new_learner_list[0])):
    Ypred.append(int(np.mean(new_learner_list[:, each_item]) > 0.5))

    
# print("MSE_validation for Ensemble is {}".format(mean_squared_error(Y_tr, Ypred)))
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat.txt', np.vstack( (np.arange(len(Ypred)) , Ypred[:]) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');

'''
Current Final Scores:
0.62284
0.62678
0.59 %either ditch adaregressor or improve it
0.64604
'''


# In[ ]:



