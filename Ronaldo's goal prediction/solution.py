# I was working in a jupyter notebook so that's why the code looks like this

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('data.csv')


data.head()

# Getting the description of the data (count,mean,std etc)
data.describe()

# Filling the missing values by merging the columns 
data['remaining_min'] = data['remaining_min'].fillna(data['remaining_min1'])
data['power_of_shot'] = data['power_of_shot'].fillna(data['power_of_shot1'])
data['knockout_match'] = data['knockout_match'].fillna(data['knockout_match_1'])
data['remaining_sec'] = data['remaining_sec'].fillna(data['remaining_sec1'])
data['distance_of_shot'] = data['distance_of_shot'].fillna(data['distance_of_shot_1'])
data['type_of_shot'] = data['type_of_shot'].fillna(data['type_of_combined_shot'])
#data = data.drop(columns=['match_id','team_id','lat/lng','team_name','match_event_id','game_season','date_of_game','shot_id_number_1'])

# dropping the repeated columns
data = data.drop(columns = ['remaining_min1','power_of_shot1','knockout_match_1','remaining_sec1','distance_of_shot_1','type_of_combined_shot'])
# dropping the columns that may be trivial for goal prediction
data = data.drop(columns = ['shot_id_number_1','team_name','team_id','match_event_id','lat/lng'])
col = data.columns
data.head()
#data.type_of_shot.isna().sum()

# Performing winsorization on the columns that we merged
for col in cols:
    quantile1 = data[col].quantile(0.95)
    quantile2 = data[col].quantile(0.05)
    data[col] = data[col].clip(quantile2,quantile1)
data.describe()


# Filling the missing values of categorical data using back filling method
data['home_or_away'] = data['home_or_away'].fillna(method = 'bfill')
data['shot_basics'] = data['shot_basics'].fillna(method = 'bfill')
data['range_of_shot'] = data['range_of_shot'].fillna(method = 'bfill')
data['area_of_shot'] = data['area_of_shot'].fillna(method = 'bfill' )

data['match_id'] = data['match_id'].fillna(method = 'bfill' )
# data['lat/lng'] = data['lat/lng'].fillna(method = 'bfill' )
# data['match_event_id'] = data['match_event_id'].fillna(method = 'bfill' )
data['game_season'] = data['game_season'].fillna(method = 'bfill' )
data['date_of_game'] = data['date_of_game'].fillna(method = 'bfill' )
print(data.shape)


# calculating the means and medians of columns which contain numerical data for filling missing values
means = dict(data.mean())
medians = dict(data.median())
print('Means: ',means)
print('Medians: ',medians)

# filling the missing numerical data using mean and median
data['location_x'] = data['location_x'].fillna(means['location_x'])
data['location_y'] = data['location_y'].fillna(means['location_y'])
data['remaining_min'] = data['remaining_min'].fillna(means['remaining_min'])
data['power_of_shot'] = data['power_of_shot'].fillna(means['power_of_shot'])
data['knockout_match'] = data['knockout_match'].fillna(medians['knockout_match'])
data['remaining_sec'] = data['remaining_sec'].fillna(means['remaining_sec'])
data['distance_of_shot'] = data['distance_of_shot'].fillna(means['distance_of_shot'])


# label encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder  = LabelEncoder()
data['shot_basics']=  labelencoder.fit_transform(data['shot_basics'])
data['range_of_shot']=  labelencoder.fit_transform(data['range_of_shot'])
data['area_of_shot']=  labelencoder.fit_transform(data['area_of_shot'])
data['type_of_shot']=  labelencoder.fit_transform(data['type_of_shot'])


data['match_id'] = labelencoder.fit_transform(data['match_id'])
# data['lat/lng'] = labelencoder.fit_transform(data['lat/lng'])
# data['match_event_id'] = labelencoder.fit_transform(data['match_event_id'])
data['game_season'] = labelencoder.fit_transform(data['game_season'])

# Converting the date of games to day, month and year
data['day'] = pd.DatetimeIndex(data['date_of_game']).day
data['month'] = pd.DatetimeIndex(data['date_of_game']).month
data['year'] = pd.DatetimeIndex(data['date_of_game']).year

# dropping the data of game column
data = data.drop(columns = ['date_of_game'])

data['day'] = labelencoder.fit_transform(data['day'])
data['month'] = labelencoder.fit_transform(data['month'])
data['year'] = labelencoder.fit_transform(data['year'])

"""
print(x_td[0:3,:])
onehotencoder = OneHotEncoder(categorical_features =[0,2,4,5,9])
x_td= onehotencoder.fit_transform(x_td).toarray()
print(x_td[0,:])

"""

# to label the home or away team 1 home while 0 represents away splitting the column values of 'home/away' to check for '@' or 'vs.'
home_team = data['home_or_away']
home_team = list(home_team)
labels  = []
for team in home_team:
    if '@' in team.split(' '):
        labels.append(1)
    else :
        labels.append(0)
data['home_or_away'] = pd.Series(labels).astype('float64')


# separating the training data
train1 = data[data['is_goal']==1]
train2 = data[data['is_goal']==0]
print(train1.shape)
print(train2.shape)
train = train1.append(train2)

# Separating the data on which we have to predict the probabilities
test = data[data['is_goal']!=1]
test = test[test['is_goal']!=0]
print(test.shape)

test.head()

train.head()

# Plotting the histogram's for columns for exploratory purposes
data.hist(figsize = (20,15))
plt.show()

# Plotting the correlation matrix to see how features are correlated with the 'is_goal'
import seaborn as sns
correlation_matrix = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(correlation_matrix,vmax = 0.8,square = True)
plt.show()

test.describe()

train.shape

# Picking the is_goal label from the train data
y  = train.iloc[:,15].values
# dropping the is_goal column from the data
train  = train.drop(columns = ['is_goal'])
X = train.iloc[:,:].values
print(X.shape)
print(y.shape)


# standard scaling the training data. Note that dropped the first column which contains shot_id_number
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X[:,1:])

# splitting the training data to fit the model and validate it later
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
X_train.shape


# Creating a neural network for predicting the probabilites
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 16,activation='relu',input_dim=17,init='glorot_uniform'))
classifier.add(Dense(units = 8,activation='relu',init='glorot_uniform'))
classifier.add(Dense(units = 4,activation='relu',init='glorot_uniform'))
classifier.add(Dense(units = 1,activation='sigmoid',init='glorot_uniform'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the training data on the neural network classifier
classifier.fit(X_train,y_train,batch_size=256,epochs=10)

#getting the predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) * 1
y_pred

#making the confusion matrix
cm = confusion_matrix(y_test,y_pred)
cm

# getting the labels of the data on which we need to predict the probabilities
y  = test.iloc[:,15].values

# dropping the column is goal
test = test.drop(columns = ['is_goal'])
X = test.iloc[:,:].values
print(X.shape)
print(y.shape)


# standard scaling the data dropping the first column
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(X[:,1:])

# getting the probabilities
y_pred = classifier.predict(data)
y_pred 

#getting the shot_id's
shot_ids = test.iloc[:,0:1].values
shot_ids
# stacking the probabilities and shot _id's
y_num = np.hstack((shot_ids ,y_pred))
#y_num = y_num.reshape((6268,2))
y_num.shape

# writing the output to a csv file
y_nb_sol = pd.DataFrame(y_num,columns=['shot_id_number','is_goal']).to_csv('y__nn.csv')


