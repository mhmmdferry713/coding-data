import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def getresult(input, output, indices, indicess, name):
    filepath1 = input

    filepath2 = "~/Documents/coding-data-main/dataset/testdataset.csv"

    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)

    testsdn = testsdn[testsdn.notnull()]
    test = test[test.notnull()]
    testsdn = testsdn.drop_duplicates(['csum','src_ip'])


    test = test[['total_length','flags','csum','src_ip','src_port','port_no','label']]
    test = test.drop_duplicates(['csum','src_ip'])

    #test = test.drop_duplicates()
    print(len(test))
    temp = len(test)
    test = test[test.index.isin(indices)]
    print(len(testsdn))
    testsdn = testsdn[testsdn.index.isin(indicess)]
    print(len(testsdn))

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    y_sdn = testsdn['Label'].values
    y_test = test['label'].values

    y_test = le.fit_transform(y_test)
    print(le.classes_)

    print(output)
    print("----------------------------------------")
    print(classification_report(y_sdn,y_test))

    print("Accuracy "+name,metrics.accuracy_score(y_test, y_sdn)*100)
    print("Precision "+name,metrics.precision_score(y_test, y_sdn,average='micro')*100)
    print("Recall "+name,metrics.recall_score(y_test, y_sdn,average='micro')*100)
    print("f1 "+name,metrics.f1_score(y_test, y_sdn,average='micro')*100)
    print("Precision "+name,metrics.precision_score(y_test, y_sdn,average='macro')*100)
    print("Recall "+name,metrics.recall_score(y_test, y_sdn,average='macro')*100)
    print("f1 "+name,metrics.f1_score(y_test, y_sdn,average='macro')*100)
    print(confusion_matrix(y_test, y_sdn))
    temp = float(39994-float(len(indices)))
    temp = float(temp/39994)
    print("packet loss :"+str(temp*100))


np.set_printoptions(suppress=True)
id1 = np.load('indexreal-adb-50pps.npy')
id1 = id1.astype(int)
print(id1.shape)
print(id1.dtype)
print(id1)
ids1 = np.load('indexsims-adb-50pps.npy')
ids1 = ids1.astype(int)
print(ids1.shape)
print(ids1.dtype)
print(ids1)

np.set_printoptions(suppress=True)
id2 = np.load('indexreal-dtc-50pps.npy')
id2 = id2.astype(int)
print(id2.shape)
print(id2.dtype)
print(id2)
ids2 = np.load('indexsims-dtc-50pps.npy')
ids2 = ids2.astype(int)
print(ids2.shape)
print(ids2.dtype)
print(ids2)

np.set_printoptions(suppress=True)
id3 = np.load('indexreal-gnb-50pps.npy')
id3 = id3.astype(int)
print(id3.shape)
print(id3.dtype)
print(id3)
ids3 = np.load('indexsims-gnb-50pps.npy')
ids3 = ids3.astype(int)
print(ids3.shape)
print(ids3.dtype)
print(ids3)

np.set_printoptions(suppress=True)
id4 = np.load('indexreal-knn-50pps.npy')
id4 = id4.astype(int)
print(id4.shape)
print(id4.dtype)
print(id4)
ids4 = np.load('indexsims-knn-50pps.npy')
ids4 = ids4.astype(int)
print(ids4.shape)
print(ids4.dtype)
print(ids4)

np.set_printoptions(suppress=True)
id5 = np.load('indexreal-mlp-50pps.npy')
id5 = id5.astype(int)
print(id5.shape)
print(id5.dtype)
print(id5)
ids5 = np.load('indexsims-mlp-50pps.npy')
ids5 = ids5.astype(int)
print(ids5.shape)
print(ids5.dtype)
print(ids5)

np.set_printoptions(suppress=True)
id6 = np.load('indexreal-rfc-50pps.npy')
id6 = id6.astype(int)
print(id6.shape)
print(id6.dtype)
print(id6)
ids6 = np.load('indexsims-rfc-50pps.npy')
ids6 = ids6.astype(int)
print(ids6.shape)
print(ids6.dtype)
print(ids6)

np.set_printoptions(suppress=True)
id7 = np.load('indexreal-svmlin-50pps.npy')
id7 = id7.astype(int)
print(id7.shape)
print(id7.dtype)
print(id7)
ids7 = np.load('indexsims-svmlin-50pps.npy')
ids7 = ids7.astype(int)
print(ids7.shape)
print(ids7.dtype)
print(ids7)

np.set_printoptions(suppress=True)
id8 = np.load('indexreal-svmrbf-50pps.npy')
id8 = id8.astype(int)
print(id8.shape)
print(id8.dtype)
print(id8)
ids8 = np.load('indexsims-svmrbf-50pps.npy')
ids8 = ids8.astype(int)
print(ids8.shape)
print(ids8.dtype)
print(ids8)

getresult('adb-50pps.csv','adb-50pps',id1, ids1, 'adb-50pps')
getresult('dtc-50pps.csv','dtc-50pps',id2, ids2, 'dtc-50pps')
getresult('gnb-50pps.csv','gnb-50pps',id3, ids3, 'gnb-50pps')
getresult('knn-50pps.csv','knn-50pps',id4, ids4, 'knn-50pps')
getresult('mlp-50pps.csv','mlp-50pps',id5, ids5, 'mlp-50pps')
getresult('rfc-50pps.csv','rfc-50pps',id6, ids6, 'rfc-50pps')
getresult('svmlin-50pps.csv','svmlin-50pps',id7, ids7, 'svmlin-50pps')
getresult('svmrbf-50pps.csv','svmrbf-50pps',id8, ids8, 'svmrbf-50pps')
