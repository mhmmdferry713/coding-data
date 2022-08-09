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
			writer.writerow({"total_length": temp[4],
			          "flags": temp[5],
			          "csum": temp[9],
			          "src_ip": temp[10],			     
			          "src_port": temp[12],
			          "port_no": temp[14],
			          "Label": temp[15]})


data = []

with open('dtc', 'r') as file:
	data = file.readlines()
	file.close()

generatecsv("dtc.csv",data)
