# -*- coding: utf-8 -*-

## This excercise is to perform different data Wrangling activities on data, using Python.

"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

import pandas as pd

transactions = pd.DataFrame({
    'TransactionID': np.arange(10)+1,
    'TransactionDate': pd.to_datetime(['2010-08-21', '2011-05-26', '2011-06-16', '2012-08-26', '2013-06-06', 
                              '2013-12-23', '2013-12-30', '2014-04-24', '2015-04-24', '2016-05-08']).date,
    'UserID': [7, 3, 3, 1, 2, 2, 3, np.nan, 7, 3],
    'ProductID': [2, 4, 3, 2, 4, 5, 4, 2, 4, 4],
    'Quantity': [1, 1, 1, 3, 1, 6, 1, 3, 3, 4]
})

transactions

import os
path = "C:\Learn and Grow\Big Data Learning\Excercises"
os.chdir(path)
os.listdir()

df = pd.read_csv("fifa_ranking_excercise4_data.csv")

df

transactions.info()

transactions.describe()

##Q2 How many rows?

In [26]: transactions.shape[0]
transactions.shape[1]
transactions.index.values
transactions.columns.values

# Q6 Change the name of column "Quantity" to "Quant"
transactions.rename(columns = {'Quantity':'Quant'},inplace = True)
transactions.columns

## Q7 Change the name of columns ProductID and UserID to PID and UID respectively
transactions.rename(columns = {'ProductID':'PID','UserID':'UID'},inplace = True)
transactions.columns
transactions.rename(columns = {'PID':'ProductID','UID':'UserID'},inplace = True)
## Q8 Ordering the rows of a DataFrame Ascending by UID
transactions.sort_values(by='UID')

# Q9 Order the rows of transactions by TransactionID descending
transactions.sort_values(by='TransactionID', ascending=0)
# Q10 Order the rows of transactions by Quantity ascending, TransactionDate descending  
transactions.sort_values(by=['Quant','TransactionDate'],ascending=[1,0])

# Q11 Ordering the columns of a DataFrame


# Q12 Set the column order of transactions as ProductID, Quantity, TransactionDate, TransactionID, UserID
transactions=transactions[['PID','Quant','TransactionDate','TransactionID','UID']]
transactions
# Q13 Make UserID the first column of transactions
transactions=transactions[['UserID','ProductID','Quant','TransactionDate','TransactionID']]
transactions

# Q14 Extracting arrays from a DataFrame
ary=transactions.values
ary

# Q15 Get the 2nd column
transactions.iloc[:,1]

# Q16 Get the ProductID array
transactions.iloc[:,3]

# Q17 Get the ProductID array using a variable
A=[]
for i in range(10):
    A=transactions.iloc[i,3]
    print(A)

# Q18 Subset rows 1, 3, and 6   (SEE ALL ANSWERS BELOW (LINE 240))
transactions.loc[[0,2,5],:]
# Q19 Subset rows exlcuding 1, 3, and 6    
transactions.loc[(-[0,2,5]),:]
