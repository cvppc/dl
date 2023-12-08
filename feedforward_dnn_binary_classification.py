from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

dataset=pd.read_csv('Script_files\diabetes.csv')
dataset.head()

from sklearn.model_selection import train_test_split
train,test = train_test_split(dataset, test_size=0.25, random_state=0, stratify=dataset['Outcome'])
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']
train_X.head()
train_Y.head()
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_X, train_Y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(test_X, test_Y)
print('Accuracy: %.2f' % (accuracy*100))