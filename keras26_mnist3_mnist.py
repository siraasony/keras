from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print (x_train)
# print (y_train)
print (x_train.shape)
print (y_train.shape)

x_train = x_train.reshape(x_train.shape[0],28, 28)
x_test = x_test.reshape(x_test.shape[0],28, 28)
print (x_train.shape)
print (x_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape = (28,28)))   # (열, 몇 개씩 자를지)
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10))

model.summary

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, epochs=100 , batch_size=1, verbose=1, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print (acc)