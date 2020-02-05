from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print (x_train)
# print (y_train)
print (x_train.shape)
print (y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(25, input_dim =784,))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(10))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, epochs=100 , batch_size=1, verbose=1, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print (acc)