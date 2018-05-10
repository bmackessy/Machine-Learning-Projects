import pandas
from sklearn import svm
from sklearn import datasets
import csv
import numpy as np
from sklearn import model_selection


def readData():
   table = []
   with open('breast-cancer.csv') as csvDataFile:
      csvReader = csv.reader(csvDataFile)
      for row in csvReader:
         newRow = []			
         for item in row:
            newRow.append(item)
         table.append(newRow)

   return table



def svmClassify(table):
   data = np.asarray(table)

   clf = svm.SVC()

   att1 = 2
   att2 = 3
   classification = 10
    
   xs = []
   ys = []

   for row in data:
      xs.append([row[att1], row[att2]])
      ys.append(row[classification])

   # Create testing and training data with 80/20 split
   x_train, x_test, y_train, y_test = model_selection.train_test_split(xs, ys, test_size=0.2)

   clf.fit(x_train, y_train)
  
   
   correct = 0
   labels = clf.predict(x_test)
   for i in range(len(labels)):
      if labels[i] == y_test[i]:
         correct += 1

   acc = correct/len(y_test)
   return acc



def main():
   print(svmClassify(readData()))

if __name__ == "__main__":
   main()



