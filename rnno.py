import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv',sep = ',')
training_set = dataset_train.iloc[: , 1:2].values
#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_scaled = sc.fit_transform(training_set)
#Crreeating a datastructure having timesteps aas 60 1 output
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(train_scaled[i-60:i , 0])
    y_train.append(train_scaled[i , 0])
X_train,y_train = np.array(X_train),np.array(y_train)
#New datastructure
X_train = np.reshape(X_train,(1198,60,1))
#making the model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
#Initialising the regressor
regressor = Sequential()
#1
regressor.add(LSTM(units = 50 , return_sequences = True,input_shape = [60,1]))
regressor.add(Dropout(.2))
#2
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(.2))
#3
regressor.add(LSTM(units = 50 , return_sequences = True))
regressor.add(Dropout(.2))
#4
regressor.add(LSTM(units = 50)
regressor.add(Dropout(.2))