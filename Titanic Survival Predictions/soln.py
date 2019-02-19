# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the dataset
td = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')
sub = pd.read_csv('gender_submission.csv')

# Filling the missing values
td.fillna(method='bfill',inplace=True)
td.fillna(method='ffill',inplace=True)
ts.fillna(method='bfill',inplace=True)
ts.fillna(method='ffill',inplace=True)

  
#Selecting out the training and test data's dependent and independent variables
x_td = td.iloc[:,2:].values
x_ts=ts.iloc[:,1:].values
y_td=td.iloc[:,1:2].values
y_sub = sub.iloc[:,1].values
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
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
x_ts = sc_X.fit_transform(x_ts)

"""
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
cm2 = confusion_matrix(y_res,y_sub)

# Using support vector machine 
from sklearn.svm import SVC
classifier1 = SVC(kernel='rbf')
classifier1.fit(X_train,y_train)

y_pred2 = classifier1.predict(X_test)
cm3= confusion_matrix(y_test,y_pred2)
y_svc = classifier1.predict(x_ts)
cm4 = confusion_matrix(y_svc,y_sub)

# Using Naive_Bayes 
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(X_train,y_train)

y_pred3 = classifier2.predict(X_test)
cm5 = confusion_matrix(y_test,y_pred3)
y_nb = classifier2.predict(x_ts)
cm6 = confusion_matrix(y_sub,y_nb)

# Using Decision trees
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(criterion='entropy')
classifier3.fit(X_train,y_train)

y_pred4 = classifier3.predict(X_test)
y_dst = classifier3.predict(x_ts)

cm7 = confusion_matrix(y_test,y_pred4)
cm8 = confusion_matrix(y_sub,y_dst)

# Using Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier4= RandomForestClassifier(n_estimators=10,criterion='entropy')
classifier4.fit(X_train,y_train)

y_pred5 = classifier4.predict(X_test)
y_rf = classifier4.predict(x_ts)

cm9 = confusion_matrix(y_test,y_pred5)
cm10 = confusion_matrix(y_sub,y_rf)

y_num = [np.arange(892,1310,dtype=np.int64) ,y_rf]
y_num = np.transpose(y_num)

y_rf_sol = pd.DataFrame(y_num,columns=['PassengerId','Survived']).to_csv('y_rf_sol.csv')



