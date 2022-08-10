# logistic regression for feature importance
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import numpy as np
import pandas as pd
import pickle

filepath1 = '/home/leviathan/Documents/Penelitian/coding+data/dataset/traindataset.csv'
filepath2 = '/home/leviathan/Documents/Penelitian/coding+data/dataset/testdataset.csv'

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

print("Logistic Regression")
model = LogisticRegression()
# fit the model
model.fit(X_important_train, y_train)
# get importance
importance = model.coef_[0]
list_features_lr = []
# summarize feature importance
for i,v in enumerate(importance):
  if v != 0:
    list_features_lr.append(i)
  print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

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
#proses LogisticRegression
print(list_features_lr)
selected_labels = [x for i,x in enumerate(feat_labels) if i in list_features_lr]
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

#training process
dcs = DecisionTreeClassifier(max_depth=5)
print("training DTC : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
dcs.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")

mlp = MLPClassifier(hidden_layer_sizes=(1, ))
print("training MLP : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
mlp.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")

#prediction process
from sklearn.metrics import classification_report
from sklearn import metrics

y_preddcs = dcs.predict(newX_important_test)
y_predmlp = mlp.predict(newX_important_test)

print("DTC Results : ")
print(classification_report(y_test, y_preddcs))
print("Accuracy:",metrics.accuracy_score(y_test, y_preddcs))

print("___________________________")

print("MLP Results : ")
print(classification_report(y_test, y_predmlp))
print("Accuracy:",metrics.accuracy_score(y_test, y_predmlp))

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
import pickle
import os

#training process
dcs = DecisionTreeClassifier(max_depth=5)
print("training DTC : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
dcs.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")

mlp = MLPClassifier(hidden_layer_sizes=(1, ))
print("training MLP : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
mlp.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")

#prediction process
from sklearn.metrics import classification_report
from sklearn import metrics

y_preddcs = dcs.predict(newX_important_test)
y_predmlp = mlp.predict(newX_important_test)

print("DTC Results : ")
print(classification_report(y_test, y_preddcs))
print("Accuracy:",metrics.accuracy_score(y_test, y_preddcs))

print("___________________________")

print("MLP Results : ")
print(classification_report(y_test, y_predmlp))
print("Accuracy:",metrics.accuracy_score(y_test, y_predmlp))

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


#training process
dcs = DecisionTreeClassifier(max_depth=5)
print("training DTC : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
dcs.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")

mlp = MLPClassifier(hidden_layer_sizes=(1, ))
print("training MLP : ")
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
mlp.fit(newX_important_train,y_train)
now = datetime.datetime.now()
print (now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
print("___________________________")


#prediction process
from sklearn.metrics import classification_report
from sklearn import metrics

y_preddcs = dcs.predict(newX_important_test)
y_predmlp = mlp.predict(newX_important_test)

print("DTC Results : ")
print(classification_report(y_test, y_preddcs))
print("Accuracy:",metrics.accuracy_score(y_test, y_preddcs))

print("___________________________")

print("MLP Results : ")
print(classification_report(y_test, y_predmlp))
print("Accuracy:",metrics.accuracy_score(y_test, y_predmlp))
