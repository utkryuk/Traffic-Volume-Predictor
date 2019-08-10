# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 02:22:50 2019

@author: Ezone
"""

#from _future_ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
import pandas as pd

np.random.seed(1671)

dataset = pd.read_csv("Train.csv")
X = dataset.iloc[:,0:14]
Y = dataset.iloc[:,-1].values


#print(dataset.info())
def f1(S):
    if S=='None':
        return 0
    else:
        return 1

X['is_holiday'] = X.is_holiday.apply(f1)



def weather_categorical(S):
    if S=='scattered clouds':
        return 1
    elif S=='broken clouds':
        return 2
    elif S=='overcast clouds':
        return 3
    elif S=='sky is clear':
        return 4
    elif S=='few clouds':
        return 5
    elif S=='light rain':
        return 6
    elif S=='light intensity drizzle':
        return 7
    elif S=='mist':
        return 8
    elif S=='haze':
        return 9
    elif S=='fog':
        return 10
    elif S=='proximity shower rain':
        return 11
    elif S=='drizzle':
        return 12
    elif S=='moderate rain':
        return 13
    elif S=='heavy intensity rain':
        return 14
    elif S=='proximity thunderstorm':
        return 15
    elif S=='thunderstorm with light rain':
        return 16
    elif S=='proximity thunderstorm with rain':
        return 17
    elif S=='heavy snow':
        return 18
    elif S=='heavy intensity drizzle':
        return 19
    elif S=='snow':
        return 20
    elif S=='thunderstorm with heavy rain':
        return 21
    elif S=='freezing rain':
        return 22
    elif S=='shower snow':
        return 23
    elif S=='light rain and snow':
        return 24
    elif S=='light intensity shower rain':
        return 25
    elif S=='SQUALLS':
        return 26
    elif S=='thunderstorm with rain':
        return 27
    elif S=='proximity thunderstorm with drizzle':
        return 28
    elif S=='thunderstorm':
        return 29
    elif S=='Sky is Clear':
        return 30
    elif S=='very heavy rain':
        return 31
    elif S=='thunderstorm with light drizzle':
        return 32
    elif S=='light snow':
        return 33
    elif S=='thunderstorm with drizzle':
        return 34
    elif S=='smoke':
        return 35
    elif S=='shower drizzle':
        return 36
    elif S=='light shower snow':
        return 37
    elif S=='sleeet':
        return 38
    else:
        return 4
    
X['weather_description'] = X.weather_description.apply(weather_categorical)

#del X['weather_type']
print(int(X['date_time'][0][11:13]))

#del X['snow_p_h']

def tym(S):
    return int(S[11:13])
X['date_time'] = X.date_time.apply(tym)


from sklearn.preprocessing import StandardScaler
sc_r = StandardScaler()
X=sc_r.fit_transform(x)
Y=sc_r.fit_transform(Y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['weather_type'] = labelencoder_X.fit_transform(X['weather_type'])

x=X
x=np.array(x)

onehotencoder = OneHotEncoder(categorical_features = [12])
x = onehotencoder.fit_transform(x).toarray()

x=np.delete(x,0,axis=1)

test_x=pd.read_csv("Test.csv")

test_x['is_holiday'] = test_x.is_holiday.apply(f1)
test_x['weather_description'] = test_x.weather_description.apply(weather_categorical)
test_x['date_time'] = test_x.date_time.apply(tym)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
test_x['weather_type'] = labelencoder_X.fit_transform(test_x['weather_type'])

test_x=np.array(test_x)

onehotencoder = OneHotEncoder(categorical_features = [12])
test_x = onehotencoder.fit_transform(test_x).toarray()


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x, Y);
predictions = rf.predict(test_x)

errors = abs(predictions - Y)
e=sum(errors)
s=sum(Y)
print(((s-e)/e)*100)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')