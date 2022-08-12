import csv

def generatecsv(input, data):
	c = input
	with open(c, "a") as f:
		fnames = [
		  "total_length",
		  "flags",
		  "csum",
		  "src_ip",
		  "src_port",
		  "port_no",
		  "Label",
		]
		writer = csv.DictWriter(f, fieldnames=fnames)
		writer.writeheader()
		for a in data:
			temp = a.rstrip("\n")
			temp = temp.split(";")
			writer.writerow({"total_length": temp[0],
			          "flags": temp[1],
			          "csum": temp[2],
			          "src_ip": temp[3],
			          "src_port": temp[4],
			          "port_no": temp[5],
			          "Label": temp[6]})

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []

with open('adb', 'r') as file:
	data1 = file.readlines()
	file.close()
with open('dtc', 'r') as file:
	data2 = file.readlines()
	file.close()
with open('gnb', 'r') as file:
	data3 = file.readlines()
	file.close()
with open('knn', 'r') as file:
	data4 = file.readlines()
	file.close()
with open('mlp', 'r') as file:
	data5 = file.readlines()
	file.close()
with open('rfc', 'r') as file:
	data6 = file.readlines()
	file.close()
with open('svmlin', 'r') as file:
	data7 = file.readlines()
	file.close()
with open('svmrbf', 'r') as file:
	data8 = file.readlines()
	file.close()

generatecsv("adb-50pps.csv",data1)
generatecsv("dtc-50pps.csv",data2)
generatecsv("gnb-50pps.csv",data3)
generatecsv("knn-50pps.csv",data4)
generatecsv("mlp-50pps.csv",data5)
generatecsv("rfc-50pps.csv",data6)
generatecsv("svmlin-50pps.csv",data7)
generatecsv("svmrbf-50pps.csv",data8)
