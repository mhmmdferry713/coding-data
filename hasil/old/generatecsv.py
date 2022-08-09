import sys
import getopt
import logging
import numpy as np
import pickle
import os
import pandas as pd
import csv


def generatecsv(input, data):
	c = input
	with open(c, "a") as f:
		fnames = [
		  "datapath id",
		  "version",
		  "header_length",
		  "tos",
		  "total_length",
		  "flags",
		  "offset",
		  "ttl",
		  "proto",
		  "csum",
		  "src_ip",
		  "dst_ip",
		  "src_port",
		  "dst_port",
		  "port_no",
		  "Label",
		]
		writer = csv.DictWriter(f, fieldnames=fnames)
		writer.writeheader()
		for a in data:
			temp = a.rstrip("\n")
			temp = temp.split(";")
			writer.writerow({"datapath id": temp[0],
			          "version": temp[1],
			          "header_length": temp[2],
			          "tos": temp[3],
			          "total_length": temp[4],
			          "flags": temp[5],
			          "offset": temp[6],
			          "ttl": temp[7],
			          "proto": temp[8],
			          "csum": temp[9],
			          "src_ip": temp[10],
			          "dst_ip": temp[11],
			          "src_port": temp[12],
			          "dst_port": temp[13],
			          "port_no": temp[14],
			          "Label": temp[15]})


data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data9 = []
with open('svmrbf', 'r') as file:
	data1 = file.readlines()
	file.close()
with open('knn', 'r') as file:
	data2 = file.readlines()
	file.close()
with open('dtc', 'r') as file:
	data3 = file.readlines()
	file.close()
with open('rfc', 'r') as file:
	data4 = file.readlines()
	file.close()
with open('mlp', 'r') as file:
	data5 = file.readlines()
	file.close()
with open('adc', 'r') as file:
	data6 = file.readlines()
	file.close()
with open('gnb', 'r') as file:
	data7 = file.readlines()
	file.close()
with open('svmlin', 'r') as file:
	data9 = file.readlines()
	file.close()

generatecsv("svmrbf.csv",data1)
generatecsv("knn.csv",data2)
generatecsv("dtc.csv",data3)
generatecsv("rfc.csv",data4)
generatecsv("mlp.csv",data5)
generatecsv("adc.csv",data6)
generatecsv("gnb.csv",data7)
generatecsv("svmlin.csv",data9)
