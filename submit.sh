#!/usr/bin/bin
#Submit HMM Monte Carlo simulation and PCA HC analysis.

#Please ensure you have the following directory structure:
#
#Date:
#	Data:
#		bitcoin.csv
#	Png:
#	Pca_csv:
#	Pca_png:

#The program will output a dendrogram and you can select the nearest neighbor
#for bitcoin.csv as the best result. Time-series forecasts will be saved in 
#Png directory so you can visualize the result. CSV data will be stored in 
#the Data directory. 




mchmc=`python bitcoin_forecast.py`;
pca=`python bitcoin_pca.py`;
cluster=`python bitcoin_clustering.py`;

