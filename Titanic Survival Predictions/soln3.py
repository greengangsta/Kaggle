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
y_td=td.iloc[:,1:2]
y_sub = sub.iloc[:,1]

X = pd.DataFrame(np.concatenate((x_td,x_ts),axis=0),columns=['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
X=X.drop(['Name','Ticket','Cabin'],axis=1)
X['Embarked'] = X['Embarked'].fillna('N')
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
onehotencoder = OneHotEncoder(categorical_features =[0,1,3,4,6])
x_td= onehotencoder.fit_transform(x_td).toarray()



from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_td = sc_X.fit_transform(x_td)

x_ts = x_td[891:,:]
x_td = x_td[:891,:]




#Splitting the traing data into training and cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_td, y_td, test_size = 0.20)


#Scaling the input features of training and test data



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

y_num = [np.arange(892,1310,dtype=np.int64) ,y_svc]
y_num = np.transpose(y_num)

y_nb_sol = pd.DataFrame(y_num,columns=['PassengerId','Survived']).to_csv('y_svc_sol.csv')



