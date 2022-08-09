import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def getresult(input, output, indices, indicess, name):
    filepath1 = input
    #disesuaikan directory dari datates
    filepath2 = "/home/fauzi/low rate attack/testdataset.csv"

    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)

    testsdn = testsdn[testsdn.notnull()]
    test = test[test.notnull()]

    #testsdn = testsdn.drop_duplicates()
    test = test.drop_duplicates()
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
id = np.load('indexreal.npy')
id = id.astype(int)
print(id.shape)
print(id.dtype)
print(id)
ids = np.load('indexsims.npy')
ids = ids.astype(int)
print(ids.shape)
print(ids.dtype)
print(ids)
#counts = np.bincount(ids)
#print(np.where(counts > 1)[0])

getresult('/home/fauzi/low rate attack/result100pps/dtc.csv','dtc',id, ids, 'dtc')
