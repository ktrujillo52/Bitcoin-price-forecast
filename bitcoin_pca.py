#!/usr/bin/python
#Principal Component Analysis CynthiaKinnan
from __future__ import division
import numpy as np
from hmmlearn import hmm
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist 
from random import randint
from  matplotlib import style
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import KernelPCA, PCA
from random import randint
style.use("ggplot")
import numpy as np
import os
import csv
import collections
import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import math

########################
#Modify these variables#
########################

data_path = "/home/kevin/Desktop/Bitcoin/github/5_16_17/data"
png_path = "/home/kevin/Desktop/Bitcoin/github/5_16_17/pca_png"
csv_path = "/home/kevin/Desktop/Bitcoin/github/5_16_17/pca_csv"



###########################################################
#Do not modify below if you do not know what you are doing#
###########################################################

class Data:
	def get_data(self):
		"""
		Read the input data in the path and store in Dataframe
		"""
		self.path = data_path
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
		#self.data = []	#list to store csv contents
		for item in self.directories:
			self.data = []
			with open(item, "r+") as f:
				self.text = csv.reader(f)
				for row in self.text:
					self.data.append(row)
				f.close()
		return self.data		

	def get_countries(self):
		self.countries = []
		for item in self.directories:
			with open(item, "r+") as f:
				self.dat = csv.reader(f)
				for row in self.dat:
					if (row[0] not in self.countries):
						self.countries.append(row[0])
				f.close()
		return self.countries

	def exports_all(self):
		#Build exports dictionary
		self.exports = {key : [float(0)] for key in self.directories}
		years = len(self.data)
		for key, value in self.exports.items():
			for year in range(0, years - 1):
				self.exports[key].append(float(0))
		#Parse export data
		for i in self.directories:
			with open(i, "r+") as f:
				self.data = []
				self.dat = csv.reader(f)
				for k in self.dat:
					self.data.append(k)
				for row in range(len(self.data)):
					try:	
						self.exports[i][row] += float(self.data[row][1])
					except:
						print "Error"
				f.close()
		print self.exports
		#Remove non-year columns from export data
		#for key, value in self.exports.items():
		#	for i in range(0, 5):
		#		self.exports[key].pop(0)
	
	def years(self):
		self.years = []
		for label in self.data:
			self.years.append(label[0])
		#for i in range(0,5):
		#	self.years.pop(0)

	def format(self):
		#Store data in pandas dataframe
		self.frame = pd.DataFrame.from_dict(self.exports, orient='columns')
		print self.years
		self.frame.index = self.years
		#self.frame = self.frame[::-1]
		print self.frame
		return self.frame


#Parse and store data in Dataframe
data = Data()
data1 = data.get_data()
data2 = data.get_path()
data3 = data.read_csv()
data4 = data.get_countries()
data6 = data.exports_all()
data7 = data.years()
data8 = data.format()


class PCAA():
	def select(self, csv, data8):
		"""
		Read each simulation from dataframe
		"""
		self.dji = pd.DataFrame(data8[csv])
		print "dji:"+str(self.dji)
	def normalize(self, data8):
		"""
		Function for scaling the data
		"""		

		self.scale_function = lambda x: (x -x.mean()) / x.std()
		
		#Apply PCA
		pca = KernelPCA().fit(preprocessing.normalize(data8))

		#Find components
		print "Total number of components calculated by PCA:"+str(len(pca.lambdas_))

		pca.lambdas_[:10].round()
		get_we = lambda x: x/ x.sum()
		get_we(pca.lambdas_)[:10]
		print get_we(pca.lambdas_)[:10]

	def pcamethod(self, data8):
		"""
		Normalize the data and run PCA. Add data to Dataframe
		column. 
		"""
		pca = KernelPCA(n_components=1).fit(preprocessing.normalize(data8))
		self.dji['PCA_1'] = pca.transform(data8)
		print "PCA"+str(self.dji['PCA_1'])
		

	def plot(self, csv):
		"""
		Plot scaled simulation data and PCA component 1. Save the scaled csv
		data for Hierarchical Clustering. 
		"""		
		self.dji.apply(self.scale_function).plot()
		save = self.dji.apply(self.scale_function)
		os.chdir(csv_path)
		app = self.dji.apply(self.scale_function)
		app[csv].to_csv('%s' % csv)
		
		plt.legend(loc = "lower right")
		plt.xlabel('Time')
		plt.ylabel('PCA')
		plt.suptitle('Principal Component Analysis')
		os.chdir(png_path)
		
		plt.xticks(rotation=90)
		
		plt.savefig('%s.png' % csv)
		
		
for csv in data8:
	data = PCAA()
	data9 = data.select(csv, data8)
	data10 = data.normalize(data8)
	data11 = data.pcamethod(data8)
	data12 = data.plot(csv)





