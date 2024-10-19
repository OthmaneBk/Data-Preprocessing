#Import necessary Packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Import DataSet
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

#Handling Missing Data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Categorical Data
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
ct=ColumnTransformer([("onehot",OneHotEncoder(),[0])],remainder='passthrough')
X=ct.fit_transform(X)

#Splitting the Dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(X_train)
