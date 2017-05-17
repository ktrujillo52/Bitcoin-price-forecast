#!/usr/bin/python
#Hierarchical clustering and dendrogram CynthiaKinnan
from matplotlib import pyplot as plt
from matplotlib.dates import datestr2num
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist 
import pandas as pd
import numpy as np
import os
import csv
import collections


########################
#Modify these variables#
########################

pca_csv_path = "/home/kevin/Desktop/Bitcoin/simulations/MCHMC/24hour/5_16_17/pca_csv/"


###########################################################
#Do not modify below if you do not know what you are doing#
###########################################################


class Data:
	"""
	Read columns 1 and 2 from csv files in pca_csv_path and generate correlation 
	matrix for linkage.
	"""
	def get_data(self):
		print "test"
		#self.path = raw_input("Enter the path to the data directory:")
		self.path = pca_csv_path 
		return self.path

	def get_path(self):
		os.chdir(self.path)

	def read_csv(self):
		#Get list of csv's in data directory
		self.directories = []	#list to store csv files
		self.names = os.listdir(self.path)
		for item in self.names:
			self.directories.append(item)
		#Read csv's
		self.data = []	#list to store csv contents
		for item in self.directories:
			with open(item, "r+") as f:
				self.text = csv.reader(f)
				for row in self.text:
					self.data.append(row)
				f.close()
		return self.data		

	def get_countries(self):
		self.countries = []
		for item in self.directories:
			self.countries.append(item)
		return self.countries

	def imports(self):
		#Build imports dictionary	
		self.imports = {key : [] for key in self.countries}
		#Parse import data
		for i in self.directories:
			with open(i, "r+") as f:
				dat = csv.reader(f)
				for row in dat:
					try:
						self.imports[i].append(float(row[1]))
					except:
						print "Error"
		#print self.imports
		self.timeseries = pd.DataFrame.from_dict(self.imports)
		print "Timeseries:"+str(self.timeseries)
	def exports(self):
		#Build exports dictionary
		self.exports = {key : []  for key in self.countries}
		#Parse export data
		for i in self.directories:
			with open(i, "r+") as f:
				dat = csv.reader(f)
				for row in dat:
					try:
						self.exports[i].append(row[0])
					except:
						print "Error"
		self.time = []
		for key in self.exports:
			self.time.append(key)
		#self.timeseries.append(time)
	def axes(self):
		#Sort dictionaries
		SortImports = collections.OrderedDict(sorted(self.exports.items(), key=lambda t: t[1]))
		SortExports = collections.OrderedDict(sorted(self.imports.items(), key=lambda t: t[1]))
		#Imports, Exports
		self.importlist = []
		self.exportlist = []
		self.sortcountrylist = []
		for key, value in SortImports.items():
			self.importlist.append(value)
			self.sortcountrylist.append(key)

		for key,value in SortExports.items():
			self.exportlist.append(value)
		self.importlist = datestr2num(self.importlist)
		self.values = zip(self.importlist, self.exportlist)
		return self.values
	
	def cluster(self):
		correlation_matrix = self.timeseries.corr(method = 'pearson')
		self.Z = linkage(self.timeseries, 'average')
	
	def dendrogram(self):
		plt.figure()
		plt.title('Hierarchical Clustering Dendrogram')
		plt.xlabel('sample index')
		plt.ylabel('distance')
		print len(self.sortcountrylist)
		dendrogram(self.Z, leaf_rotation=90., leaf_font_size= 8., labels = self.sortcountrylist)
		plt.show()

data = Data()
data1 = data.get_data()
data2 = data.get_path()
data3 = data.read_csv()
data4 = data.get_countries()
data5 = data.imports()
data6 = data.exports()
data7 = data.axes()
data9 = data.cluster()
data10 = data.dendrogram()
	
