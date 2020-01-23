# data
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_val = np.array([101,102,103,104,105])
y_val = np.array([101,102,103,104,105])

x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])


# print (x.shape)
# print (y.shape)

# model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# training
model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train,validation_data= (x_val, y_val), epochs=100, batch_size=1)


# predict
mse, acc = model.evaluate(x_test, y_test, batch_size=1)
print ('acc:', acc, 'mse:', mse)

# x_pred = np.array([11,12,13])
# aaa = model.predict(x_pred, batch_size=1)
# print (aaa)

# bbb = model.predict(x, batch_size=1)
# print (bbb)
