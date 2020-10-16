import numpy as np
import pandas as pd
import scipy
#from loaddata import data
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("diabetic1.csv", None)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

from sklearn.linear_model import LogisticRegression   
# create logistic regression object 
reg = LogisticRegression() 
   
# train the model using the training sets 
reg.fit(x_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(x_test) 

w = reg.coef_ 
# comparing actual response values (y_test) with predicted response values (y_pred) 
#print(w)
print("Logistic Regression model accuracy(in %):",  
metrics.accuracy_score(y_test, y_pred)*100) 

