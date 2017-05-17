#!/usr/bin/python
from __future__ import division
import numpy as np
from hmmlearn import hmm
from matplotlib.dates import datestr2num
from matplotlib import pyplot as plt
from mpldatacursor import datacursor
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist 
from random import randint
from  matplotlib import style
from random import randint
from itertools import izip
style.use("ggplot")
import datetime
import numpy as np
import os
import csv
import collections
import warnings
import itertools
import pandas as pd
import statsmodels.api as sm
import math
import matplotlib.dates as mdates

#############################
#Modify only these variables#
#############################

data_path = "/home/kevin/Desktop/Bitcoin/github/5_16_17/data/" 		#path to data directory
png_path = "/home/kevin/Desktop/Bitcoin/github/5_16_17/png/"		#path to png directory
simulations = 24													#default number of simulations, should not be greater than the number of samples in dataset
iterations = 24														#Number of steps to predict (hours)
start = datetime.datetime(2017, 5, 16, 8,0,0)						#this should be the last date/time in your dataset
finish = datetime.datetime(2017, 5, 17, 8,0,0)						#start variable plus the number of iterations
interval = datetime.timedelta(hours = 1)							#hours, days, months, years



###########################################################
#Do not modify below if you do not know what you are doing#
###########################################################
class Data:
	def get_data(self):
		"""
		Read path to data directory. There should be a csv
		file in the directory with price data. Columns 1
		and 2 from csv file will be extracted into array
		self.l 
		"""
		warnings.filterwarnings("ignore")
		#print "test"
		#self.path = raw_input("Enter the path to the data directory:")
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
		self.data = []	#list to store csv contents
		for item in self.directories:
			with open(item, "r+") as f:
				self.text = csv.reader(f)
				for row in self.text:
					self.data.append(row)
				f.close()
		#print self.data
		return self.data	#contains csv data	


	def bitcoindata(self):
		#Build 
		time = []
		price = []
		for i in self.data:
			time.append(i[0])
			price.append(i[1])
		self.time = time
		self.price = price	

	def timeseries(self):
		#select = raw_input("Enter country for MCHMC forecast:")
		#set x axis 
		self.x = self.price
		#set y axis
		self.y = self.time
		#Data
		self.l = zip(self.y, self.x)		
		#Plot
		#plt.plot(self.y, self.x)
		#plt.show()
		return self.l

	
#Call Data class methods
data = Data()
data1 = data.get_data()
data2 = data.get_path()
data3 = data.read_csv()
data5 = data.bitcoindata()
data8 = data.timeseries()
data8.pop(0)

class HMM():	

	def transition(self):
		"""
		Calculate transition probabilities for the first Hidden
		Markov Model. Algorithm counts how many times the input
		data goes up or down and calculates probability 
		frequencies.
		"""

		self.data = data8
		print self.data
		count_down = 0
		count_up = 0
		self.export_data = []
		self.years = []
		self.originalexport = []
		#self.data.pop(0)
		#self.data.pop(-1)
		
		for key, value in self.data:
			self.export_data.append(float(value))
			self.years.append(key)
			self.originalexport.append(float(value))
		print self.export_data
		print self.years
		for export in range(0, len(self.export_data) - 1):
			diff = float(self.export_data[export + 1]) - float(self.export_data[export])
			if (diff >= 0):
				count_up += 1
			if (diff < 0):
				count_down += 1	
		self.transition_probability = [count_down // len(self.export_data), count_up // len(self.export_data)]
		
	def transition2(self):
		"""
		Calculate transition probabilities for the second Hidden
		Markov Model. Algorithm uses standard deviation as a metric
		to count the number of high and low price changes and calculates
		probability frequencies.
		"""
		self.sd = np.std(self.export_data)		
		count_high = 0
		count_low = 0		
		for price in range(0, len(self.export_data)-1):
			diff = float(self.export_data[price + 1]) - float(self.export_data[price])
			if (diff >= self.sd / 2):
				count_high += 1
			if (diff < self.sd / 2):
				count_low += 1
		self.transition_probability2 = [count_low // len(self.export_data), count_high // len(self.export_data)]

	def model(self):
		"""
		Initialize Hidden Markov Model with initial probabilities
		start_probability and transition probabilities transition_probability.
		Uses Viterbi algorithm to calculate most probable path and generates
		a sample to learn the parameters (emission probability) using Gaussian. 
		"""

		states = ["Down", "Up"]
		n_states = len(states)
                                                                    
		observations = ["down down", "down up", "up down", "up up"]
		n_observations = len(observations)
                                                 
		start_probability = np.array([1 / 2, 1 / 2])

		#Learn parameters of model using fit
		model = hmm.GaussianHMM(n_components = n_states, covariance_type = "diag" ,startprob_prior = start_probability, transmat_prior = np.array(self.transition_probability), params = "s")
		array = []
		for i in self.export_data:
			array.append(i)

		#print np.array([self.export_data]).T
		model.fit(np.array([self.export_data]).T)
		self.X, Z = model.sample(len(self.export_data))
		#print self.X
		#print Z
		
		
		#Determine most probable state sequence
		model2 = hmm.GaussianHMM(n_states, "diag", ) #startprob_prior = start_probability, transmat_prior = np.array(self.transition_probability))
		new = []
		for i in self.X:
			new.append(i)	
		model2.fit(new)
		self.Z2 = model2.predict(self.X)
		visible = np.array([self.export_data]).T
		#print "Hidden states:"+str(len(self.Z2))
		hidden = model2.decode(visible, algorithm="viterbi")

		#print "Hidden states:", ", ".join(map(lambda x: states[x], hidden))

	def model2(self):
		"""
        Initialize Hidden Markov Model with initial probabilities
        start_probability and transition probabilities transition_probability2.
        Uses Viterbi algorithm to calculate most probable path and generates
        a sample to learn the parameters (emission probability) using Gaussian. 
		"""
		states = ["Low", "High"]
		n_states = len(states)
                                                                    
		observations = ["down down", "down up", "up down", "up up"]
		n_observations = len(observations)
                                                 
		start_probability = np.array([1 / 2, 1 / 2])

		#Learn parameters of model using fit
		model = hmm.GaussianHMM(n_components = n_states, covariance_type = "diag" ,startprob_prior = start_probability, transmat_prior = np.array(self.transition_probability2), params = "s")
		array = []
		for i in self.export_data:
			array.append(i)

		#print np.array([self.export_data]).T
		model.fit(np.array([self.export_data]).T)
		self.X, Z = model.sample(len(self.export_data))
		#print self.X
		#print Z
		
		
		#Determine most probable state sequence
		model2 = hmm.GaussianHMM(n_states, "diag", ) #startprob_prior = start_probability, transmat_prior = np.array(self.transition_probability))
		new = []
		for i in self.X:
			new.append(i)	
		model2.fit(new)
		self.Z3 = model2.predict(self.X)
		visible = np.array([self.export_data]).T
		#print "Hidden states:"+str(len(self.Z3))
		hidden = model2.decode(visible, algorithm="viterbi")

		#print "Hidden states:", ", ".join(map(lambda x: states[x], hidden))

	def montecarlo(self):
		"""
		Run Monte Carlo simulation. Uses random standard deviation guided by 
		4 states total. 
		States:
			[Up, High]
			[Up, Low]
			[Down, High]
			[Down, Low]
		The output cannot be less than zero because the Hidden State will switch
		to Up and generate a positive number. 
		"""
		if (self.Z2[-1] == 1):	#up
			if (self.Z3[-1] == 1):
				self.export_data.append(self.export_data[-1] + randint(int(round(math.sqrt(self.sd))), int(round(self.sd))))
			if (self.Z3[-1] == 0):
				self.export_data.append(self.export_data[-1] + randint(0, int(round(self.sd / 2))))
		if (self.Z2[-1] == 0):
			if (self.Z3[-1] == 1):                                                                   			
				if (float(self.export_data[-1]) >= 0):
					print self.export_data[-1]
					self.export_data.append(self.export_data[-1] - randint(int(round(math.sqrt(self.sd))), int(round(self.sd))))
				else:
					self.export_data.append(self.export_data[-1] + randint(math.sqrt(self.sd), int(round(self.sd))))
			if (self.Z3[-1] == 0):
				if (float(self.export_data[-1]) >= 0):
					self.export_data.append(self.export_data[-1] - randint(0, int(round(self.sd / 2))))
				else:
					self.export_data.append(self.export_data[-1] + randint(0, int(round(self.sd / 2))))
		data = self.export_data
		return data

	def plot(self, iterations, filecount):
		print self.years
		print len(self.years)
		print len(self.originalexport)
		dates = datestr2num(self.years)

		a = plt.plot(dates, self.originalexport, 'r-', label = 'Original')

		count = start 
		self.time = []
		end = finish 
		step = interval 
		while count < end:
			self.time.append(count.strftime('%Y-%m-%d %H:%M:%S'))
			count += step

		for i in range(0, len(self.years) + 1):
			self.export_data.pop(0)
		seed = datetime.datetime(2017, 5, 16, 8, 0, 0)
		
		self.time = self.time[::-1]
		self.time.append(seed.strftime('%Y-%m-%d %H:%M:%S'))
		self.time = self.time[::-1]
	
		self.export_data = self.export_data[::-1]
		self.export_data.append(self.originalexport[-1])
		self.export_data = self.export_data[::-1]
		
		self.time.pop(0) 
		
		dates = self.time
		dates = datestr2num(dates)

		b = plt.plot(dates, self.export_data, 'b-', label = 'Forecast')

		plt.legend(loc = 'lower right')
		plt.xlabel('Date')
		
		plt.ylabel('Bitcoin Price')
		 
		plt.suptitle('Monte Carlo Hidden Markov Model Bitcoin Price Forecast')
	
		#Display plot
		#plt.show()

		#Save plot
		os.chdir(png_path)
		plt.savefig('%s.png' % filecount)
		l = zip(self.time, self.export_data)
		plt.clf()
		del self.export_data
		return	l
#Run HMM and Monte Carlo, predict outcome at each iteration. 
filecount = 0
while (filecount < simulations):	
	hmm0 = HMM()
	
	iterations = iterations
	hmm1 = hmm0.transition()
	hmm2 = hmm0.transition2()
	for i in range(0, iterations):			
		hmm3 = hmm0.model()
		hmm4 = hmm0.model2()
		hmm5 = hmm0.montecarlo()

	#Plot results
	forecast = hmm0.plot(iterations, filecount)
	
	#Save to csv

	class SAVE():
		def tocsv(self, filecount):
				os.chdir(data_path)
				with open('%s.csv' % filecount, 'wb') as f:		
					writer = csv.writer(f)
					array1 = []
					array2 = []
					for key, value in forecast:
						array1.append(key)
						array2.append(value)
					writer.writerows(izip(array1, array2))
				f.close()

	send = SAVE()
	send1 = send.tocsv(filecount)
	filecount += 1
