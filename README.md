# Bitcoin-price-forecast
2 Hidden Markov Models are used to guide Monte Carlo simulation. The data is analyzed with Principal Component Analysis and Hierarchical Clustering. The best forecast will be the nearest-neighbor of 'bitcoin.csv', or the input data, from the Hierarchical Clustering analysis. 
## Requirements
Python
## Usage
Please ensure you have the following directory structure and replace the path variables at the top of each python script:
```
Date
|-- data                #contains input data and will contain csv data generated from HMM and Monte Carlo simulation
    |-- bitcoin.csv     #input data
|-- png                 #plots generated from HMM and Monte Carlo simulation
|-- pca_csv             #Contains scaled simulation data
|-- pca_png             #Principal Component Analysis plots
```
To submit a job please ensure all scripts are in the same directory and run the following command:
```
bash submit.sh
```

A dendrogram will appear and the nearest-neighbor to 'bitcoin.csv' will be the best forecast from the simulations. 

## License
This program is licensed under the terms of the MIT license. See [LICENSE.txt](https://github.com/ktrujillo52/Bitcoin-price-forecast/blob/master/LICENSE.txt) for more information. 

## Questions
Please forward any questions or troubleshooting issues to k.trujillo52@gmail.com
