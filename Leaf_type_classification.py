# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:40:19 2017

@author: Diwas.Tiwari
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import keras

data_path = 'D:\Diwas\Leaf_classification/train.csv'
df = pd.read_csv(data_path)
print (df.info())
print (df.head(n = 5))

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
## Seprating input and output features ##
features = df.drop(['species','id'], axis = 1)
target = df.species
## Label Encoding Textual Fetaures ##
encode = LabelEncoder()
fit = encode.fit(target)
target_enc = fit.transform(target)
## normalizing the pixel intensity values ##
#train = train/255
##For Data Standardization ##
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
fit = standard.fit(features, y = None)
features_trans = fit.transform(features)

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout

target_enc_hot_encoded = to_categorical(target_enc)
x_train,x_test,y_train,y_test = train_test_split(features_trans,target_enc_hot_encoded, train_size = 0.8 )
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
## Building the model ##
model = Sequential()
model.add(Dense(512, activation = 'relu', input_dim = 192)) ## "input_dim" for extracted features and input_shape for "input_images" ##
model.add(BatchNormalization(axis = -1))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 20, epochs = 150, 
                    validation_data = (x_test, y_test))
[loss,accuracy] = model.evaluate(x_test, y_test)

print('Output of model training are: {}'.format(loss, accuracy))

## plotting loss and accuracy ##
plt.figure(figsize = [7,7])
plt.plot(history.history['acc'], color = 'red')
plt.plot(history.history['val_acc'], color = 'cyan')
plt.title('Accuracy Graph for Leaf Classification')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Training_Acc', 'Validation_Acc'])

plt.figure(figsize = [7,7])
plt.plot(history.history['loss'], color = 'green')
plt.plot(history.history['val_loss'], color = 'magenta')
plt.title('Loss Graph for Leaf Classification')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['Training_Loss', 'Validation_Loss'])

####################################                        

## Building the second model ##
model = Sequential()
model.add(Dense(512, activation = 'relu', input_dim = 192)) ## "input_dim" for extracted features and input_shape for "input_images" ##
model.add(BatchNormalization(axis = -1))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization(axis = -1))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(99, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 20, epochs = 100, 
                    validation_data = (x_test, y_test))
[loss,accuracy] = model.evaluate(x_test, y_test)

print('Output of model training are: {}'.format(loss, accuracy))

## plotting loss and accuracy ##
plt.figure(figsize = [7,7])
plt.plot(history.history['acc'], color = 'red')
plt.plot(history.history['val_acc'], color = 'cyan')
plt.title('Accuracy Graph for Leaf Classification')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Training_Acc', 'Validation_Acc'])

plt.figure(figsize = [7,7])
plt.plot(history.history['loss'], color = 'green')
plt.plot(history.history['val_loss'], color = 'magenta')
plt.title('Loss Graph for Leaf Classification')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['Training_Loss', 'Validation_Loss'])

                         

####################################
