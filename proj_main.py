import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from logisticClassify2 import *
import sklearn.ensemble.AdaBoostClassifier as adaboost
from sklearn import tree

np.random.seed(0)

X_test = np.genfromtxt("data/X_test.txt",delimiter=None)
X = np.genfromtxt("data/X_train.txt",delimiter=None)
Y = np.genfromtxt("data/Y_train.txt",delimiter=None)

X,Y = ml.shuffleData(X,Y)
X = X[:20000]
Y = Y[:20000]
[Xtr,Xva,Ytr,Yva] = ml.splitData(X,Y,0.50)

learner_list = []

##K-Nearest
print("K-Nearest Neighbor")
k_list=[5, 55, 100, 1000, 3000]
learner = []
##use 55
for k in range(len(k_list)): #Figure out this error
	knn = ml.knn.knnClassify(Xtr, Ytr, k_list[k]) # create the object and train it
	learner.append(knn)
	auc = knn.auc(Xva, Yva)
	print("AUC score for k={} is {}.".format(k_list[k], auc))
list_55 = learner[1]
learner_list.append(list_55)

for k in range(len(learner)):
	learner[k].predictSoft(X_test)
	train_roc = learner[k].roc(Xtr, Ytr)
	valid_roc = learner[k].roc(Xva, Yva)
	plt.plot(valid_roc[0],valid_roc[1])
	plt.plot(valid_roc[1],valid_roc[1])
	plt.show()

###Linear models.
###use logistic classify
#print("Linear models")
#lr_model = logisticClassify2(); # create and train model
# xs = np.linspace(0,10,200); # densely sample possible x-values
# xs = xs[:,np.newaxis] # force "xs" to be an Mx1 matrix (expected by our code)
# ys = lr_model.predict( xs ); # make predictions at xs

# training_mse = lr_model.mse(Xtr, Ytr)
# valid_mse = lr_model.mse(Xte, Yte)


###Random forest. 



###Final Ensemble Combo
ensemble = adaboost()
ensemble.estimators_ = learner_list
Ypred = ensemble.predict(X_test)


'''
#Final Prediction
###Ypred = learner.predictSoft(X_test)   # make "soft" predictions from your learner  (Mx2 numpy array)

# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat.txt', np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T, '%d, %.2f',header='ID,Prob1',comments='',delimiter=',');
'''