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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)


#Scaling the input features of training and test data

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
x=sc_X.transform(x)

y_train=y_train.reshape((39900,1))
y_test = y_test.reshape((2100,1))

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features =[0])
y_train= onehotencoder.fit_transform(y_train).toarray()
y_test = onehotencoder.fit_transform(y_test).toarray()


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 512,activation='relu',input_dim=784,init='glorot_uniform'))

classifier.add(Dense(units = 512,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 256,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 256,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 64,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 64,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 32,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 16,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units = 10,activation='softmax',init='glorot_uniform'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=128,epochs=50)

y_pred_ann = classifier.predict(X_test)

y_pred_ann = np.argmax(y_pred_ann,axis=-1)
y_test = np.argmax(y_test,axis=-1)

y_ann = classifier.predict(x)
y_ann = np.argmax(y_ann,axis=-1)

cmann = confusion_matrix(y_test,y_pred_ann)

y_num=np.column_stack((np.arange(1,28001,dtype=np.int64) ,y_ann))
y_ann_sol = pd.DataFrame(y_num,columns=['ImageId','Label']).to_csv('y_ann_sol2.csv')







