# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')
sub = pd.read_csv('gender_submission.csv')

#Selecting out the training and test data's dependent and independent variables
x_td = td.iloc[:,2:]
x_ts=ts.iloc[:,1:]
y_td=td.iloc[:,1:2].values
y_sub = sub.iloc[:,1]

X = pd.DataFrame(np.concatenate((x_td,x_ts),axis=0),columns=['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
X=X.drop(['Name','Ticket','Cabin'],axis=1)
X['Embarked']= X['Embarked'].fillna('S')
x_td = X.iloc[:,:].values


#Filling the missing values through imputer
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values ='NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(x_td[:,2:3])
x_td[:,2:3] = imputer.transform(x_td[:,2:3])
imputer = imputer.fit(x_td[:,5:6])
x_td[:,5:6] = imputer.transform(x_td[:,5:6])


#Label and one-hot encoding the training data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X  = LabelEncoder()
x_td[:,1]=  labelencoder_X.fit_transform(x_td[:,1])
x_td[:,6]=  labelencoder_X.fit_transform(x_td[:,6])
onehotencoder = OneHotEncoder(categorical_features =[0,1,6])
x_td= onehotencoder.fit_transform(x_td).toarray()


print(x_td[0:2,:])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_td = sc_X.fit_transform(x_td)

x_ts = x_td[891:,:]
x_td = x_td[:891,:]




#Splitting the traing data into training and cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_td, y_td, test_size = 0.10)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=16,activation='relu',init='glorot_uniform',input_dim=12))

classifier.add(Dense(units=8,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units=4,activation='relu',init='glorot_uniform'))

classifier.add(Dense(units=1,activation='sigmoid',init='glorot_uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=32,epochs=2000)


from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.50)*1
cm4 = confusion_matrix(y_test,y_pred)



# Making the confusion matrix 


# Using support vector machine 
from sklearn.svm import SVC
classifier1 = SVC(kernel='rbf')
classifier1.fit(X_train,y_train)

y_pred2 = classifier1.predict(X_test)
cm3= confusion_matrix(y_test,y_pred2)
y_svc = classifier1.predict(x_ts)



y_num = [np.arange(892,1310,dtype=np.int64) ,y_svc]
y_num = np.transpose(y_num)

y_nb_sol = pd.DataFrame(y_num,columns=['PassengerId','Survived']).to_csv('y_svc3_sol.csv')



