import sys
import getopt
import logging
import numpy as np
import pickle
import os
import pandas as pd

def getindex(input, output, outputs):
    filepath1 = input
    #disesuaikan directory dari datates
    filepath2 = "/home/fauzi/low rate attack/testdataset.csv"


    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)

    testsdn = testsdn[testsdn.notnull()]
    test = test[test.notnull()]

    testsdn = testsdn.drop_duplicates()
    test = test.drop_duplicates()
    print(len(testsdn))
    print(len(test))


    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    X_sdn = testsdn.drop(columns=['Label'])
    y_sdn = testsdn['Label'].values

    X_test = test.drop(columns=['rx_bytes_ave','rx_error_ave','rx_dropped_ave','tx_bytes_ave','tx_error_ave','tx_dropped_ave','label'])
    y_test = test['label'].values

    y_test = le.fit_transform(y_test)

    data = np.array([])
    datatemp = np.array([])
    i = 0

    for a in X_sdn.index:
        index_list = X_test[(X_test['datapath_id']==X_sdn['datapath id'][a])&(X_test['version']==X_sdn['version'][a])&(X_test['header_length']==X_sdn['header_length'][a])&(X_test['tos']==X_sdn['tos'][a])&(X_test['total_length']==X_sdn['total_length'][a])&(X_test['flags']==X_sdn['flags'][a])&(X_test['offset']==X_sdn['offset'][a])&(X_test['ttl']==X_sdn['ttl'][a])&(X_test['proto']==X_sdn['proto'][a])&(X_test['csum']==X_sdn['csum'][a])&(X_test['src_ip']==X_sdn['src_ip'][a])&(X_test['dst_ip']==X_sdn['dst_ip'][a])&(X_test['src_port']==X_sdn['src_port'][a])&(X_test['dst_port']==X_sdn['dst_port'][a])&(X_test['port_no']==X_sdn['port_no'][a])].index.tolist()

        if(len(index_list)==1):
            datatemp = np.append(datatemp,int(a))
            data = np.append(data,int(index_list[0]))
        else:
            i=i+1
            print(i)


    print(data)
    np.save(output, data)
    np.save(outputs, datatemp)

#directory file csv result disesuaikan
getindex('/home/fauzi/low rate attack/result100pps/dtc.csv','indexreal.npy','indexsims.npy')
