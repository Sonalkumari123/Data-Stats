# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:08:31 2018

@author: Sonal Kumari
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#print('File location: %s' % filelocation)
 #Reading csv data from file
data= pd.read_csv(r'F:\ML&NN_codes/ex1data1.csv');
#data.head()
data.info()


#Segregating training data & test data
print('Number of data set :%s'%len(data))
trainingData= data[:(int(0.7*len(data)))]
length_of_testdata= len(data)-len(trainingData)
testingData= data[:length_of_testdata]
print('length of training data = %s' % len(trainingData))
print(trainingData)
print('length of test data = %s' % length_of_testdata)
print(testingData)
Number_of_column_data= len(data.columns);
print('Number of columns in data set: %s' % Number_of_column_data)
datavar=data.columns.values.tolist()
print(datavar)
print('-----------------------')

#Training Data segregation based on x & y
X = trainingData.iloc[:,0:(Number_of_column_data-1)];
print('Training Data set varible(x) part:')
print(X)
Y = trainingData.iloc[:,Number_of_column_data-1];
print('Training Data set output(Y) :')
print(Y)
if(Number_of_column_data==2):
 X= X.values.reshape(-1,1);

#setting object for linear regression
regr = linear_model.LinearRegression();

# Train the model using the training sets
regr.fit(X, Y);
model_coef= regr.coef_;
model_intercept= regr.intercept_;
print("Model Coefficients: %s"% model_coef)
print("Model Intercept:%s"% model_intercept)

#Prepare testing data for model 
testX=testingData.iloc[:,0:(Number_of_column_data-1)];
#print('Testing Data set varible(x) part:')
#print(testX)
testY = testingData.iloc[:,Number_of_column_data-1];
#print('Testing Data set output(Y) :')
#print(testY)

#calculating the predicted Y value from testing x data( testX)
pred_y= regr.predict(testX);
#print("Predicted values:")
#print(pred_y);
print('====================================================')
#The mean squared error of model & R^2 data for predicted model
print("Mean squared error: %.2f" % mean_squared_error(testY,pred_y))
print('====================================================')
# Explained variance score: 1 is perfect prediction
print('R^2 score of the Model : %.2f' % r2_score(testY, pred_y) , ' (1 is perfect fit model, -ve value give worst fitting)')
print('====================================================')

#Predicted linear model
##Deprecating the weights to 2 decimal point
model_intercept_d= format(model_intercept, '.2f')
model_coef_d=[]
model= ''
for k in range(0,len(model_coef)):
    model_coef_d.append(format(model_coef[k], '.2f'))

for j in range(0,len(model_coef_d)):
    model+=' +'
    model+= ' (' + model_coef_d[j] +')'
    model+='* '
    model+=datavar[j]
   

#print(model_intercept_d)
#print(model_coef_d)
print('---------Best Fit Model based on given data-------------')
print('    '+datavar[Number_of_column_data-1]+' ='+' ('+model_intercept_d+')'+model)
print('========================================================')

# Plot outputs
if(Number_of_column_data==2):
  print("visual interpretation of model & data shown below:")
  plt.scatter(X, Y,  color='black')
  plt.scatter(testX, testY,  color='red')
  plt.plot(testX, pred_y, color='blue', linewidth=2)

  plt.xticks(())
  plt.yticks(())
  plt.show()

  
    
def prediction(sampledata):
    outcome=regr.predict(sampledata);
    print('Prediction output: %s' % outcome)
    
def predictionf(filelocation):
    print('File location: %s' % filelocation)
    pdata= pd.read_csv(filelocation);
    print(pdata)
    outcome=regr.predict(pdata);
    print('Prediction output: %s' % outcome)
    df = pd.DataFrame(data=outcome)
    print(df)
   # df.to_csv(filelocation,sep='\t', encoding='utf-8' )
   