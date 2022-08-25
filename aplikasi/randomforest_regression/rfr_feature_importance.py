from matplotlib import pyplot
import numpy as np
import pandas as pd
import pickle

filepath1 = '/home/aybe/Desktop/coding+data/dataset/traindataset.csv'
filepath2 = '/home/aybe/Desktop/coding+data/dataset/testdataset.csv'

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
print("RF-Regressor")
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# fit the model
model.fit(X_important_train, y_train)
# get importance
importance = model.feature_importances_
list_features_rfr = []
# summarize feature importance
for i,v in enumerate(importance):
    if v != 0:
        list_features_rfr.append(i)
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


###################################################################
print("")
print("-----------------------------------------------------------------------------")

#proses RF-Regressor

print(list_features_rfr)
selected_labels = [x for i,x in enumerate(feat_labels) if i in list_features_rfr]
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
from sklearn.metrics import classification_report
from sklearn import metrics


import pickle
import os

filename = "svmlin.sav"
myFile = open(filename, "w+")
myFile.close()
clfl = SVC(kernel="linear", C=0.025)
print("1 svmlinear")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
clfl.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(clfl, open(filename, 'wb'))


filename = "svmrbf.sav"
myFile = open(filename, "w+")
myFile.close()
clfr = SVC(kernel="rbf", C=0.025)
print("2 svmrbf")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
clfr.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(clfr, open(filename, 'wb'))



filename = "rfc.sav"
myFile = open(filename, "w+")
myFile.close()
rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
print("3 rfc")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
rfc.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(rfc, open(filename, 'wb'))


filename = "dtc.sav"
myFile = open(filename, "w+")
myFile.close()
dcs = DecisionTreeClassifier(max_depth=5)
print("4 dtc")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
dcs.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(dcs, open(filename, 'wb'))


filename = "mlp.sav"
myFile = open(filename, "w+")
myFile.close()
mlp = MLPClassifier(hidden_layer_sizes=(1, ))
print("5 mlp")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
mlp.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(mlp, open(filename, 'wb'))

filename = "gnb.sav"
myFile = open(filename, "w+")
myFile.close()
gnb = GaussianNB()
print("6 gnb")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
gnb.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(gnb, open(filename, 'wb'))

filename = "adb.sav"
myFile = open(filename, "w+")
myFile.close()
adb = AdaBoostClassifier(n_estimators=100, random_state=0)
print("7 adb")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
adb.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(adb, open(filename, 'wb'))

# for i in range(1,20):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(newX_important_train,y_train)
#     y_preddcs = knn.predict(newX_important_test)
#     print("knn-"+str(i)+" Results : ")
#     print(classification_report(y_test, y_preddcs))
#     print("Accuracy:",metrics.accuracy_score(y_test, y_preddcs))
#     print("___________________________")


filename = "knn.sav"
myFile = open(filename, "w+")
myFile.close()
knn = KNeighborsClassifier(n_neighbors=3)
print("8 knn")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
knn.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")
pickle.dump(knn, open(filename, 'wb'))


#prediction process

y_predclfl = clfl.predict(newX_important_test)
y_predclfr = clfr.predict(newX_important_test)
y_predrfc = rfc.predict(newX_important_test)

y_preddcs = dcs.predict(newX_important_test)
y_predmlp = mlp.predict(newX_important_test)

y_predgnb = gnb.predict(newX_important_test)
y_predadb = adb.predict(newX_important_test)
y_predknn = knn.predict(newX_important_test)

print("SVMLIN Results : ")
print(classification_report(y_test, y_predclfl))
print("Accuracy:",metrics.accuracy_score(y_test, y_predclfl))

print("___________________________")

print("SVM RBF Results : ")
print(classification_report(y_test, y_predclfr))
print("Accuracy:",metrics.accuracy_score(y_test, y_predclfr))

print("___________________________")

print("RFC Results : ")
print(classification_report(y_test, y_predrfc))
print("Accuracy:",metrics.accuracy_score(y_test, y_predrfc))

print("___________________________")

print("DTC Results : ")
print(classification_report(y_test, y_preddcs))
print("Accuracy:",metrics.accuracy_score(y_test, y_preddcs))

print("___________________________")

print("MLP Results : ")
print(classification_report(y_test, y_predmlp))
print("Accuracy:",metrics.accuracy_score(y_test, y_predmlp))

print("___________________________")

print("GNB Results : ")
print(classification_report(y_test, y_predgnb))
print("Accuracy:",metrics.accuracy_score(y_test, y_predgnb))

print("___________________________")

print("ADB Results : ")
print(classification_report(y_test, y_predadb))
print("Accuracy:",metrics.accuracy_score(y_test, y_predadb))

print("___________________________")

print("KNN Results : ")
print(classification_report(y_test, y_predknn))
print("Accuracy:",metrics.accuracy_score(y_test, y_predknn))

print("___________________________")


###################################################################
print("")
print("-----------------------------------------------------------------------------")
