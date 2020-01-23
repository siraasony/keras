# data
import numpy as np
x = np.array([range(1,101), range(101, 201)])
y = np.array([range(1,101), range(101, 201)])
print (x.shape)
print (y.shape)

x = np.transpose(x)
y = np.transpose(y)
# x = np.reshape(x, (100,2))
# y = np.reshape(y, (100,2))
print (x.shape)
print (y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66,shuffle = False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=66, shuffle = False)

# model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape =(2,)))
model.add(Dense(128))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(2))

model.summary()

# training
model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, validation_data= (x_val, y_val), epochs=100, batch_size=1)

# predict
mse, acc = model.evaluate(x_test, y_test, batch_size=1)
print ('acc:', acc, 'mse:', mse)

x_pred = np.array([[201,202,203],[204,205,206]])
x_pred = np.transpose(x_pred)
aaa = model.predict(x_pred, batch_size=1)
print (aaa)

y_predict = model.predict(x_test, batch_size = 1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print ("RMSE :", RMSE(y_test, y_predict))


# bbb = model.predict(x, batch_size=1)
# print (bbb)

# R2구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print ("R2:", r2_y_predict)
