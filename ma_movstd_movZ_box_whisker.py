# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:36:18 2019
@author: Sonal Kumari

To call this function usee the raw string data. put 'r' in front of string data
 ma(filelocation,days,zlimit)

1. Sample function call using default moving z score where days=7,zlimit=3  
 
#  ma(r'C:\HPP\Desktop\data100.csv')

2. Sample function call using days=5 & z score = +2 & -2   
#  ma(r'C:\HPP\Desktop\data100.csv',5,2)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import re  as reg
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

z=[]
def mza(movA,stD,Y):
 global z
 for i in range(0, len(Y)):
   z.append((Y[i]-movA[i])/ stD[i]);
 print('\n => Moving Zscore: ')
 print(z) 
 return 0  

def average(data):
    print(data)
    sum=0
    l= len(data)
    for i in range(0,l):
        sum+= data[i]
    print('Sum :%s,'%sum)
    avrg= sum/l
    print('-----------------')
    print('Average: %s:'%avrg) 
    return avrg

def standardD(data):
    standdard_dev= np.std(data)
    print('std deviation of this set of data %s'%standdard_dev)
    print('===================================================')
    return standdard_dev

def ma(filelocation,days=7,zlimit=3):

#Created on Sun Jan 13 15:36:18 2019
#@author: Sonal Kumari

#this function takes the 'file location' as input , you can also give the numbers of days taken in to 
#account for calculating the moving average. By default( if no argument is passed) it will calculate
#for 7 days , & zlimit=3 
#1. Sample function call using default days (i.e. 7)  
#  ma(r'C:\HPP\Desktop\data100.csv')

#2. Sample function call using parameterised days(i.e 6)  
#  ma(r'C:\HPP\Desktop\data100.csv',6,2)


#setting moving average set up
 print('\n=> File location: %s' % filelocation)

#Reading csv data from file
 data= pd.read_csv(filelocation);
#classifying data 
 Y = data['yval'];
 date=data['date']; 
 mi= len(Y)-days+1
 print('\n=> i(number of full sets of yval) used for the iterator : %s'%mi)
 print('\n => number of partial set of yval :%s'%(days-1))
 print('-------')
 avg=[]
 std=[]
 ydata=[]
 global z
 z=[]
 daysby2=int(days/2)
 print(Y)
 print('\n\n=====================================')
 
# for upside incomplete dataset 
 for up in range (0,daysby2):
    ydata.clear();
    for m in range(0,(up+daysby2+1)):
      ydata.append(Y[m])
    print('ydata for i :%s,'%up)
    avg.append(average(ydata))
    std.append(standardD(ydata))

#for middle complete dataset    
 if(days%2 != 0 or days%2==0):
  for k in range(0,mi):
   ydata.clear()
   for l in range(0+k,days+k):
       ydata.append(Y[l])
   print('ydata for i :%s,'%(l-2))
   avg.append(average(ydata))
   std.append(standardD(ydata))  
   
#for lower incomplete dataset    
 for lo in range((len(Y)-daysby2),(len(Y))):
    ydata.clear();
    for m in range(lo-daysby2,(len(Y))):
      ydata.append(Y[m])
    print('ydata for i :%s,'%lo)
    #print(ydata)
    avg.append(average(ydata))
    std.append(standardD(ydata))
 print('-----------------------------------------')
 print('=============+++++++++++++++++======================')
 print('=============+++++++++++++++++======================')
 print('\n=> Moving Average : ')
 print(avg)
 print('\n=> Moving Standard Deviation : ')
 print(std)  
    
#moving average engine
 #zS=[]
 print('-----------------------------------------')  
 print('*  Length of data column(Y) :%s'% len(Y)) 
 print('*  Length of output moving average array :%s'% len(avg))
 print('*  Length of output moving standard deviation array :%s'% len(std))
 mza(avg,std,Y)
 
 
#Preprocessing the date format::
 month=[]
 zero=[]
 print(date)
 print(type(date[0]))
 print(len(date))
 for i in range(len(date)):
     month.append(i+1)
     zero.append(0)
 zero=np.array(zero)
      
 
#plotting engine:::::::::==================>>>>>>>>>>>
 
 fig,ax = plt.subplots(1)
 plt.title("Original data Plot (Y value vs days)")
 plt.xlabel("days")
 plt.ylabel("yval");
 plt.scatter(date, Y, color='black', marker='+')
 plt.plot(date,Y,color='green',linewidth=1)
 ax.set_xticklabels(month)
 plt.show()


 #below 3 lines - plotting the moving average data
 #ax = fig.add_subplot(111)
 #for n in range(0,len(avg)):
 #   ax.annotate(' %0.1f'%avg[n], xy=(date[n],avg[n]))
 fig,ax = plt.subplots(1)
 plt.title("Moving average Plot")
 plt.xlabel("Days Progression -->")
 plt.ylabel("Moving Average with days: %s"%days);
 plt.plot(date,avg, color='red',linewidth=2) 
 ax.set_xticklabels(month)
 #Below line is for the 'x' marker at moving avg
 plt.scatter(date, avg, color='blue', marker='x')
 plt.show()
 
 
 fig,ax = plt.subplots(1)
 plt.title("Moving average Plot overlapped with data plot")
 plt.xlabel("Days Progression -->")
 plt.ylabel("Moving Average ; original data ");
 plt.plot(date,Y, color='green',linewidth=0.5, label="Original Data") 
 plt.plot(date,avg, color='red',linewidth=2,label="moving average") 
 plt.legend()
 ax.set_xticklabels(month)
 plt.show()
 
 
 fig,ax = plt.subplots(1)
 plt.title("Moving Avg,  Std deviation &  data plot")
 plt.xlabel("days")
 plt.ylabel("MA; Std Dev; Original data");
 plt.plot(date,Y, color='green',linewidth=0.5,label="original Data")
 plt.plot(date,avg, color='red',linewidth=2,label="moving average") 
 plt.plot(date,std, color='blue',linewidth=1,label="Standard deviation") 
 plt.legend()
 ax.set_xticklabels(month)
 plt.show()
 
 
 fig,ax = plt.subplots(1)
 plt.title("Z score Plot")
 plt.xlabel("days")
 plt.ylabel("Z score");
 plt.plot(date,zero+zlimit, color='blue',linestyle='dashed', linewidth=1)   #
 plt.plot(date,(zero-zlimit), color='blue', linestyle='dashed',linewidth=1)
 plt.plot( date,zero, color='black', linewidth=1)
 plt.scatter(date,z,  color='blue',marker='+')
 for i in range(0,len(date)):
  if(z[i]>zlimit or z[i]<-zlimit):
    plt.scatter(date[i],z[i],  color='red',marker='x')
    ax.annotate('%s'%date[i], xy=(date[i],z[i]))
 ax.set_xticklabels(month)
 plt.show()
 
#CSV data writing engine(it will generate the movingavg.csv file to folder where this code is stored)

 df1=pd.DataFrame(data[['date','yval']])
 df1['movA=%s'%days] = pd.Series(avg)
 df1['movStD=%s'%days] = pd.Series(std)
 df1['movZscore=%s'%days]= pd.Series(z)
 print('***   Final output data:')
 print(df1)
 df1.to_csv('movingA_STD_Z1.csv', encoding='utf-8', index=False)
 
 print('------------------------------\nData Statistics of Y:')
 print( Y.describe())
 #boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])
 green_diamond = dict(markerfacecolor='r', marker='D')
 fig1, ax1 = plt.subplots()
 ax1.set_title('Box and Whiskers Plot and Outliers')
 ax1.boxplot(Y,flierprops=green_diamond, vert=False,notch=True, whis=1.5)