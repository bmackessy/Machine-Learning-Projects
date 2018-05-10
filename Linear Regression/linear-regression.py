# Part 1 - Linear Regression

import csv
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import datasets, linear_model
from sklearn import cross_validation


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



# Xs represent Uniformity of Cell Size: 1 - 10
# Ys represent Uniformity of Cell Shape: 1 - 10
def getXsYs(table):
   xs = []
   ys = []
   for row in table:
      xs.append(int(row[2]))    # 2 is index of cell size
      ys.append(int(row[3]))    # 3 is index of cell shape 
   
   return np.array(xs), np.array(ys)



# formula for best fit line
def best_fit_slope_and_intercept(xs,ys):
   m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))
   b = mean(ys) - m*mean(xs)
	
   return m, b



# Validating the regression
def coefficient_of_determination(ys_orig,ys_line):
   y_mean_line = [mean(ys_orig) for y in ys_orig]
	
   # Other usable error metrics
   squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
   squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

   r_squared = 1 - (squared_error_regr/squared_error_y_mean)

   return r_squared



# Homemade version of sklearn.LinearRegression()
# Question 1, Part 1
def q1p1():
   mTable = readData()
   xs, ys = getXsYs(mTable)
   
   # Create testing and training data with 80/20 split
   x_train, x_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.2)

   m, b = best_fit_slope_and_intercept(x_train, y_train)
   
   # Re-creating each y-value for x test values 
   regression_line = [(m*x)+b for x in x_test]
  
   r_squared = coefficient_of_determination(y_test,regression_line)
   
   print()
   print()
   print()
   print()
   print("Homemade Linear Regression")
   print('r^2 = ' + str(r_squared))
   print('m   = ' + str(m))
   print('b   = ' + str(b))
   print()
   plt.scatter(x_test,y_test,color='#003F72', label = 'data')
   plt.plot(x_test, regression_line, label = 'homemade')
   plt.legend(loc=4)
   plt.savefig('plot1.png',format='png')



# Using sklean.LinearRegression() uses least-squares
# Question 1, Part 2
def q1p2():
   mTable = readData()
   xs, ys = getXsYs(mTable)
   
   # Create LinearRegression object
   regr = linear_model.LinearRegression()
   
   # Create testing and training data with 80/20 split
   x_train, x_test, y_train, y_test = cross_validation.train_test_split(xs, ys, test_size=0.2)

   # Train the linear regression model
   regr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
   
   m = regr.coef_
   b = regr.intercept_
  
   # Calculate a r^2 (coefficient of determination)
   r_squared = regr.score(x_test.reshape(-1,1), y_test.reshape(-1,1))


   # Re-creating each y-value for x test values 
   regression_line = np.asarray([(m*x)+b for x in x_test]).reshape(-1,1)
   
   print("Using Sklearn Linear Regression")
   print("r^2 = " + str(r_squared))
   print('m   = ' + str(m))
   print('b   = ' + str(b))

   plt.scatter(x_test,y_test,color='#003F72', label = 'data')
   plt.plot(x_test, regression_line, label = 'builtin')
   plt.legend(loc=4)
   plt.savefig('LinearRegression.png',format='png')
   
   

# main
q1p1()
q1p2()
