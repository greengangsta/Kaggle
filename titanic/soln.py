import numpy as np
import pandas as pd

td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')

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
#x_td[:,8]=  labelencoder_X.fit_transform(x_td[:,8])
x_td[:,9]=  labelencoder_X.fit_transform(x_td[:,9])
print(x_td[0,:])
onehotencoder = OneHotEncoder(categorical_features =[0])
x_td= onehotencoder.fit_transform(x_td).toarray()
labelencoder_y = LabelEncoder()
# y_td= labelencoder_y.fit_transform(y)
