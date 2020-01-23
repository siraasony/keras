# data
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
y_train = y[:60]
x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:]
y_val = y[80:]


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

x_pred = np.array([101,102,103])
aaa = model.predict(x_pred, batch_size=1)
print (aaa)

# bbb = model.predict(x, batch_size=1)
# print (bbb)
