import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')

td.fillna(method='bfill',inplace=True)
td.fillna(method='ffill',inplace=True)

  

x_td = td.iloc[:,2:].values
x_test=ts.iloc[:,2:].values
y_td=td.iloc[:,1:2].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values ='NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(x_td[:,3:4])
x_td[:,3:4] = imputer.transform(x_td[:,3:4])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X  = LabelEncoder()
x_td[:,1]=  labelencoder_X.fit_transform(x_td[:,1])
x_td[:,0]=  labelencoder_X.fit_transform(x_td[:,0])
x_td[:,6]=  labelencoder_X.fit_transform(x_td[:,6])
x_td[:,2]=  labelencoder_X.fit_transform(x_td[:,2])
x_td[:,9]=  labelencoder_X.fit_transform(x_td[:,9])
x_td[:,8]=  labelencoder_X.fit_transform(x_td[:,8])
print(x_td[0:3,:])
onehotencoder = OneHotEncoder(categorical_features =[0,2,4,5,9])
x_td= onehotencoder.fit_transform(x_td).toarray()
print(x_td[0,:])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_td, y_td, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)



