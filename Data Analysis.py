import numpy as np
import time
import pandas as pd
import scipy
import os
import matplotlib.pyplot as plot
from scipy import signal
import seaborn as sns

# 1. Load the datasets into numpy objects
np.set_printoptions(suppress=True) #genfrom text now will give int values rather than scientific notation
original_messidor = np.genfromtxt('messidor_features.txt', skip_header = 24, delimiter=',')

# 2. make a panda object to use it in task2 and try in linear regration
def load_data(path, header):
    df = pd.read_csv(path, delimiter=',',header=22)
    return df
if __name__ == "__main__":
# load the data from the file
    data = load_data("messidor_features.txt", None)
    print(data)

# 3. Do not need to clean the data because there arent any missing values

# 4. Perform some statistics
#show histogram of target col
targetCol = original_messidor[:,-1]
histogramOfTargetColumn= plot.hist(targetCol, color='red')
plot.show() 
#correlation of the credit_Card_Data
dataForCorrelation = original_messidor[:,:-1] #everything but the last column
corr = data.corr()
sns.heatmap(corr)
plot.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.95:
            print(i,j)
            
        else:
            #print(i)
            continue
#Correlation with output variable
cor_target = abs(corr.iloc[:,-1])#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

fig, a = plot.subplots(3,3)
a[0][0].scatter(data.iloc[:,2], data.iloc[:,3])
a[0][1].scatter(data.iloc[:,2], data.iloc[:,4])
a[0][2].scatter(data.iloc[:,2], data.iloc[:,5])
a[1][0].scatter(data.iloc[:,3],data.iloc[:,4])
a[1][1].scatter(data.iloc[:,3],data.iloc[:,5])
a[1][2].scatter(data.iloc[:,4],data.iloc[:,5])
a[2][0].scatter(data.iloc[:,4],data.iloc[:,6])
a[2][1].scatter(data.iloc[:,5],data.iloc[:,6])
a[2][2].scatter(data.iloc[:,6],data.iloc[:,7])


plot.show()

sns.distplot(data.iloc[:,2])
sns.distplot(data.iloc[:,7])
plot.show()

sns.boxplot(data=data.iloc[:,2])
plot.show()
