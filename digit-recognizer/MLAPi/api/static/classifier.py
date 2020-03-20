import numpy as np
import pandas as pd
import cv2
import warnings 
warnings.filterwarnings('ignore')

# matplotlib.pyplot as plt
"""
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

classifier.fit(X_train,y_train,batch_size=128,epochs=10)


# code for saving and loading the model

from keras.models import model_from_json

classifier_json = classifier.to_json()

with open('classifier.json','w') as json_file:
	json_file.write(classifier_json)
	
classifier.save_weights('classifier.h5')
"""
import pickle
img = cv2.imread('test')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(784,1))
f = open('scaler.txt','rb')
scaler = pickle.loads(f.read())
img = scaler.transform(img)

from keras.models import model_from_json
json_file = open('classifier.json','r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)
loaded_classifier.load_weights('classifier.h5')
loaded_classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
pred = loaded_classifier.predict(img)

print("Executed Successfully",np.argmax(pred,axis= -1))

"""


y_pred_ann = classifier.predict(X_test)

y_pred_ann_loaded = loaded_classifier.predict(X_test)

loaded_classifier.fit(X_train,y_train,batch_size=128,epochs=20)

y_pred_ann = np.argmax(y_pred_ann,axis=-1)
y_test = np.argmax(y_test,axis=-1)

y_ann = classifier.predict(x)
y_ann = np.argmax(y_ann,axis=-1)

cmann = confusion_matrix(y_test,y_pred_ann)

y_num=np.column_stack((np.arange(1,28001,dtype=np.int64) ,y_ann))
y_ann_sol = pd.DataFrame(y_num,columns=['ImageId','Label']).to_csv('y_ann_sol2.csv')


"""
