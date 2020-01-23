# data
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print (x.shape)
# print (y.shape)

# model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim =1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

model.add(Dense(1))

# training
model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)

# predict
mse, acc = model.evaluate(x, y, batch_size=1)
print ('acc:', acc, 'mse:', mse)

# x_pred = np.array([11,12,13])
# aaa = model.predict(x_pred, batch_size=1)
# print (aaa)

# bbb = model.predict(x, batch_size=1)
# print (bbb)