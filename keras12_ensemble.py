#레이어의 깊이 -> 노드의 갯수 -> 

# data
import numpy as np
x = np.array([range(1,101), range(101, 201), range(301, 401)])
x2 = np.array([range(1001, 1101), range(1101,1201), range(1301, 1401)])
y1 = np.array([range(101, 201)])

#y2 = np.array([range(1101, 1201)])

# print (x.shape) # (3, 100)
# print (y.shape) # (1, 100)

x1 = np.transpose(x) #(100, 3)
x2 = np.transpose(x)
y1 = np.transpose(y1) #(100, 1)


# x = np.reshape(x, 300)
# y = np.reshape(y, 100)
# print (x.shape)
# print (y.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2, y1, train_size=0.6, random_state=66,shuffle = False)
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(x1_test, x2_test, y1_test, test_size=0.5, random_state=66, shuffle = False)
# x2_train, x2_val, y1_train, y1_val = train_test_split(x2_train, y1_train, test_size=0.25, random_state=66, shuffle = False)
print (x1_train.shape, y1_val.shape, x2_val.shape)
       
# model
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(3,))
dense21 = Dense(7)(input2)
dense23 = Dense(4)(dense21)
output2 = Dense(5)(dense23)

from keras.layers.merge import concatenate

from keras.layers import Concatenate

#merge1 = concatenate([output1, output2]) #모델을 사슬처럼 엮다.
merge1 = Concatenate()([output1, output2])
#class와 function형 concatenate의 차이 -  import

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2], outputs = output)
#두개 이상 넣을 때는 list형식으로 넣으면 된다! / 1개도 list 무방


#model.add(Dense(5))
# model.add(Dense(2))
# model.add(Dense(3))
# model.add(Dense(1)) #should change the output shape!!!!!!!!!

model.summary()

# training
model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])
model.fit([x1_train,x2_train], y1_train, validation_data= ([x1_val, x2_val], y1_val), epochs=100, batch_size=30)

# predict
mse, acc = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print ('acc:', acc, 'mse:', mse)

x1_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])
x1_pred = np.transpose(x1_pred)
x2_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])
x2_pred = np.transpose(x2_pred)
aaa = model.predict([x1_pred, x2_pred], batch_size=1)
print (aaa)

y_predict = model.predict([x1_test, x2_test], batch_size = 1)
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y_predict))
print ("RMSE :", RMSE(y1_test, y_predict))


# bbb = model.predict(x, batch_size=1)
# print (bbb)

# R2구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y_predict)
print ("R2:", r2_y_predict)
