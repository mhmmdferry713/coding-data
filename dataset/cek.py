import pandas as pd

test = pd.read_csv('testdataset.csv')
temp = test[test.duplicated(['csum','src_ip'])]
print(temp)
