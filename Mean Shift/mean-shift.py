import pandas as pd
import csv
import numpy as np
from sklearn import cluster
from sklearn.cluster import MeanShift



def mean_shift():
   table = pd.read_csv("breast-cancer-wisconsin.data", header=0)
   table.replace('?',-99999, inplace=True)
   table.drop(table.columns[0], axis=1, inplace=True)

   clf = MeanShift()
   clf.fit(table)

   labels = clf.labels_
   cluster_centers = clf.cluster_centers_
   
   labels_unique = np.unique(labels)
   
   n_clusters = len(labels_unique)
        
   centroids = cluster_centers
   return n_clusters
   

print(mean_shift())

