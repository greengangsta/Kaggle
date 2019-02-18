# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')

# Filling the missing values
td.fillna(method='bfill',inplace=True)
td.fillna(method='ffill',inplace=True)
ts.fillna(method='bfill',inplace=True)
ts.fillna(method='ffill',inplace=True)

  
#Selecting out the training and test data's dependent and independent variables
x_td = td.iloc[:,2:].values
x_ts=ts.iloc[:,1:].values
y_td=td.iloc[:,1:2].values
print(x_td[0,:])
print(x_ts[0,:])

#Filling the missing values through imputer
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values ='NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(x_td[:,3:4])
x_td[:,3:4] = imputer.transform(x_td[:,3:4])
imputer = imputer.fit(x_ts[:,3:4])
x_ts[:,3:4] = imputer.transform(x_ts[:,3:4])



#Label and one-hot encoding the training data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X  = LabelEncoder()
x_td[:,1]=  labelencoder_X.fit_transform(x_td[:,1])
x_td[:,0]=  labelencoder_X.fit_transform(x_td[:,0])
x_td[:,6]=  labelencoder_X.fit_transform(x_td[:,6])
x_td[:,2]=  labelencoder_X.fit_transform(x_td[:,2])
x_td[:,9]=  labelencoder_X.fit_transform(x_td[:,9])
x_td[:,8]=  labelencoder_X.fit_transform(x_td[:,8])
print(x_td[0:3,:])
onehotencoder = OneHotEncoder(categorical_features =[0,2,9])
x_td= onehotencoder.fit_transform(x_td).toarray()
print(x_td[0,:])

#Label and one hot encoding on the test data
labelencoder_X  = LabelEncoder()
x_ts[:,1]=  labelencoder_X.fit_transform(x_ts[:,1])
x_ts[:,0]=  labelencoder_X.fit_transform(x_ts[:,0])
x_ts[:,6]=  labelencoder_X.fit_transform(x_ts[:,6])
x_ts[:,2]=  labelencoder_X.fit_transform(x_ts[:,2])
x_ts[:,9]=  labelencoder_X.fit_transform(x_ts[:,9])
x_ts[:,8]=  labelencoder_X.fit_transform(x_ts[:,8])
print(x_ts[0:3,:])
onehotencoder = OneHotEncoder(categorical_features =[0,2,9])
x_ts= onehotencoder.fit_transform(x_ts).toarray()
print(x_ts[0,:])

#Splitting the traing data into training and cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_td, y_td, test_size = 0.25, random_state = 0)


#Scaling the input features of training and test data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
x_ts = sc_X.fit_transform(x_ts)

#Training the logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_res = classifier.predict(x_ts)


# Making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)






