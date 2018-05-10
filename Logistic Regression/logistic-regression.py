# Part 2 - Logistic Regression

import pandas as pd
from statsmodels.discrete import discrete_model
import math
import numpy as np
from sklearn import cross_validation
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import style


def q2():
   print()
   print()
   print()

   data = pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")

   # Turn string attribute values into 0 or 1
   dummy_ranks = pd.get_dummies(data["Sex"], prefix="Sex")

   # Save class labels
   classLabels = data["Survived"]
   
   # Remove attributes that don't help with classification or has been turned into an int already (sex)
   data = data.drop(["PassengerId","Name","Ticket", "Cabin", "Embarked", "Sex", "Survived"], 1)
   data = data.join(dummy_ranks.ix[:'Sex_female'])

   # Create training and testing data (80/20 split)
   x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, classLabels, test_size=0.2)

   # Create and train logistic model
   logit = discrete_model.Logit(y_train, x_train, missing='drop')
   fitted = logit.fit()
   
   # Predict testing data
   result = fitted.predict(x_test)
   
   result = np.asarray(result)
   y_test = np.asarray(y_test)
   
   total = 0
   correct = 0
   for i in range(0, len(result)):
      if not math.isnan(result[i]):
         total += 1
         if round(result[i]) == y_test[i]:
            correct += 1
   
   accuracy = correct/float(total)
   print("Accuracy of Logistic Regression: " + str(accuracy))



def model(x):
    return 1 / (1 + np.exp(-x))

q2()
   





