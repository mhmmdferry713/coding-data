import numpy as np
import pandas as pd

def getindex(input, output, outputs):
    filepath1 = input

    filepath2 = "~/Desktop/coding+data/dataset/testdataset.csv"

    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)
    #test = pd.read_csv(filepath2, usecols=['total_length','flags','csum','src_ip','src_port','port_no','label'])

    testsdn = testsdn[testsdn.notnull()]
    test = test[test.notnull()]

    #testsdn = testsdn.drop_duplicates()
    #test = test.drop_duplicates()
    print(len(testsdn))
    print(len(test))


    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    X_sdn = testsdn.drop(columns=['Label'])

    X_sdn = X_sdn.drop_duplicates(['csum','src_ip'])


    #X_test = test.drop(columns=['datapath_id', 'version', 'header_length', 'tos', 'offset', 'ttl', 'proto', 'dst_ip', 'dst_port'])
    #X_test = test[['total_length','flags','csum','src_ip','src_port','port_no','label']]
    X_test = test[['total_length','flags','csum','src_ip','src_port','port_no','label']]
    X_test = X_test.drop_duplicates(['csum','src_ip'])
    print(X_test)
    y_test = test['label'].values

    y_test = le.fit_transform(y_test)

    data = np.array([])
    datatemp = np.array([])
    i = 0

    for a in X_sdn.index:
        index_list = X_test[(X_test['total_length']==X_sdn['total_length'][a])&(X_test['flags']==X_sdn['flags'][a])&(X_test['csum']==X_sdn['csum'][a])&(X_test['src_ip']==X_sdn['src_ip'][a])&(X_test['src_port']==X_sdn['src_port'][a])&(X_test['port_no']==X_sdn['port_no'][a])].index.tolist()

        if(len(index_list)==1):
            datatemp = np.append(datatemp,int(a))
            data = np.append(data,int(index_list[0]))
        else:
            i=i+1
            print(i)

    print(data)
    np.save(output, data)
    np.save(outputs, datatemp)

getindex('~/Desktop/coding+data/hasil_final/pps50/adb-50pps.csv','indexreal-adb-50pps.npy','indexsims-adb-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/dtc-50pps.csv','indexreal-dtc-50pps.npy','indexsims-dtc-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/gnb-50pps.csv','indexreal-gnb-50pps.npy','indexsims-gnb-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/knn-50pps.csv','indexreal-knn-50pps.npy','indexsims-knn-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/mlp-50pps.csv','indexreal-mlp-50pps.npy','indexsims-mlp-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/rfc-50pps.csv','indexreal-rfc-50pps.npy','indexsims-rfc-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/svmlin-50pps.csv','indexreal-svmlin-50pps.npy','indexsims-svmlin-50pps.npy')
getindex('~/Desktop/coding+data/hasil_final/pps50/svmrbf-50pps.csv','indexreal-svmrbf-50pps.npy','indexsims-svmrbf-50pps.npy')
