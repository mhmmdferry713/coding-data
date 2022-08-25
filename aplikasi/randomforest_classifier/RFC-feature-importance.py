from matplotlib import pyplot
import numpy as np
import pandas as pd
import pickle

filepath1 = '/home/ubuntu/Documents/coding-data-main/dataset/traindataset.csv'
filepath2 = '/home/ubuntu/Documents/coding-data-main/dataset/testdataset.csv'

train = pd.read_csv(filepath1)
test = pd.read_csv(filepath2)

train = train[train.notnull()]
test = test[test.notnull()]

train.head()
test.head()

feat_labels = list(train.columns)

print("")
print(feat_labels)

print("")
print(len(feat_labels))

print("")
print(len(train))

print("")
print(len(test))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

X_train = train.drop(columns=['label'])
y_train = train['label'].values
# print(y_train)

X_train = X_train.apply(le.fit_transform)
y_train = le.fit_transform(y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_important_train = scaler.fit_transform(X_train)

print("")
print("RF-Classifier")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# fit the model
model.fit(X_important_train, y_train)
# get importance
importance = model.feature_importances_
list_features_rfc = []
# summarize feature importance
for i,v in enumerate(importance):
    if v != 0:
        list_features_rfc.append(i)
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

###################################################################
print("")
print("-----------------------------------------------------------------------------")
#proses RF-Classifier

print(list_features_rfc)
selected_labels = [x for i,x in enumerate(feat_labels) if i in list_features_rfc]
print(selected_labels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

newX_train = train[selected_labels]
print(newX_train.head())
y_train = train['label'].values

newX_test = test[selected_labels]
y_test = test['label'].values

newX_train = newX_train.apply(le.fit_transform)
y_train = le.fit_transform(y_train)

newX_test = newX_test.apply(le.fit_transform)
y_test = le.fit_transform(y_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
newX_important_train = scaler.fit_transform(newX_train)
newX_important_test = scaler.fit_transform(newX_test)

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import datetime


import pickle
import os

def rumus_ges (filename, order, nowo1):
    myFile = open(filename, "w+")
    myFile.close()
    nowo = nowo1
    print(order)
    now = datetime.datetime.now()
    print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    nowo.fit(newX_important_train,y_train)
    now = datetime.datetime.now()
    print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print("_________")
    pickle.dump(nowo, open(filename, 'wb'))

clfl = SVC(kernel="linear", C=0.025)
rumus_ges('svmlin.sav' , '1 svmlinear', clfl)

clfr = SVC(kernel="rbf", C=0.025)
rumus_ges('svmrbf.sav' , '2 svmrbf', clfr)

rfc=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rumus_ges('rfc.sav' , '3 rfc', rfc)

dcs=DecisionTreeClassifier(max_depth=5)
rumus_ges('dtc.sav' , '4 dtc', dcs)

mlp=MLPClassifier(hidden_layer_sizes=(1, ))
rumus_ges('mlp.sav' , '5 mlp', mlp)

gnb=GaussianNB()
rumus_ges('gnb.sav' , '6 gnb', gnb)

adb=AdaBoostClassifier(n_estimators=100, random_state=0)
rumus_ges('adb.sav' , '7 adb', adb)

knn=KNeighborsClassifier(n_neighbors=3)
rumus_ges('knn.sav' , '8 knn', knn)

from sklearn.metrics import classification_report
from sklearn import metrics

def pred_ges(ypred1 , nama):
    y_pred = ypred1
    print(nama)
    print(classification_report(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


pred_ges(clfl.predict(newX_important_test) , "CLFL Results : ")
pred_ges(clfr.predict(newX_important_test) , "clfr Results : ")
pred_ges(rfc.predict(newX_important_test) , "rfc Results : ")
pred_ges(dcs.predict(newX_important_test) , "dcs Results : ")
pred_ges(mlp.predict(newX_important_test) , "mlp Results : ")
pred_ges(gnb.predict(newX_important_test) , "gnb Results : ")
pred_ges(adb.predict(newX_important_test) , "adb Results : ")
pred_ges(knn.predict(newX_important_test) , "knn Results : ")
