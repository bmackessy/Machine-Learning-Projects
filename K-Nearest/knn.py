# Part 3 - K-Nearest Neighbors

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
from sklearn import neighbors 



# Homemade KNN function
def k_nearest_neighbors(data, predict, k):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
  
    return vote_result, confidence



# Uses homemade KNN classifer
# Question 3, Part 1
def q4p1():
   df = pd.read_csv("breast-cancer-wisconsin.data.txt")
   df.replace('?',-99999, inplace=True)
   df.drop(['id'], 1, inplace=True)
   full_data = df.astype(float).values.tolist()
   random.shuffle(full_data)

   test_size = 0.4
   train_set = {2:[], 4:[]}
   test_set = {2:[], 4:[]}
   train_data = full_data[:-int(test_size*len(full_data))]
   test_data = full_data[-int(test_size*len(full_data)):]

   for i in train_data:
      train_set[i[-1]].append(i[:-1])
    
   for i in test_data:
      test_set[i[-1]].append(i[:-1])

   correct = 0
   total = 0

   for group in test_set:
      for data in test_set[group]:
         vote,confidence = k_nearest_neighbors(train_set, data, k=4)
         if group == vote:
            correct += 1
         total += 1

   print('Accuracy of homemade KNN:', correct/float(total))



# Builtin KNN Classifier
# Question 4, Part 2
def q4p2():
   df = pd.read_csv("breast-cancer-wisconsin.data.txt")
   df.replace('?',-99999, inplace=True)
   df.drop(['id'], 1, inplace=True)
   full_data = df.astype(float).values.tolist()
   random.shuffle(full_data)

   test_size = 0.4
   train_set = {2:[], 4:[]}
   test_set = {2:[], 4:[]}
   train_data = full_data[:-int(test_size*len(full_data))]
   test_data = full_data[-int(test_size*len(full_data)):]

   for i in train_data:
      train_set[i[-1]].append(i[:-1])
    
   for i in test_data:
      test_set[i[-1]].append(i[:-1])
   
   trainingDataX = train_set[2] + train_set[4]
   trainingDataY = []
   for i in range(len(train_set[2])):
      trainingDataY.append(2)
   for i in range(len(train_set[4])):
      trainingDataY.append(4)

   testingDataX = test_set[2] + test_set[4]
   testingDataY = []
   for i in range(len(test_set[2])):
      testingDataY.append(2)
   for i in range(len(test_set[4])):
      testingDataY.append(4)

   classifier = neighbors.KNeighborsClassifier(n_neighbors=4)
   trainedClassifier = classifier.fit(trainingDataX, trainingDataY)
   
   predictions = trainedClassifier.predict(testingDataX)

   correct = 0
   total = 0

   for i in range(len(testingDataY)):
      total += 1
      if predictions[i] == testingDataY[i]:
         correct += 1
  
   print("Accuracy of builin KNN  : " + str(correct/float(total)))


q4p1()
q4p2()
