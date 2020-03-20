# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')
  
#Selecting out the training and test data's dependent and independent variables
X=td.iloc[:,1:].values
y=td.iloc[:,0].values
x=ts.iloc[:,:].values


#Splitting the traing data into training and cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Scaling the input features of training and test data

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
x=sc_X.transform(x)

import pickle
#scaler_string = pickle.dumps(sc_X)
#open('sclaer.txt','wb').write(scaler_string)
f = open('scaler.txt','rb')
scaler = pickle.loads(f.read())
type(scaler)

from sklearn.svm import SVC
classifier1 = SVC(kernel='linear')
classifier1.fit(X_train,y_train)


from sklearn.metrics import confusion_matrix
y_pred = classifier1.predict(X_test)
cm= confusion_matrix(y_test,y_pred)

y_svc = classifier1.predict(x)


y_num=np.column_stack((np.arange(1,28001,dtype=np.int64) ,y_svc))
y_ann_sol = pd.DataFrame(y_num,columns=['ImageId','Label']).to_csv('y_svc_sol.csv')











