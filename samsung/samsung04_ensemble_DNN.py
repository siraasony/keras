import numpy as np
import pandas as pd

samsung = np.load('./data/samsung.npy')
kospi200 = np.load('./data/kospi200.npy')

print (kospi200)
print (kospi200.shape)


def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)
# print (x.shape)
# print (y.shape)
# print (x[0,:], '\n', y[0])

# ================================묶음으로 잘라내기================================#

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 1, test_size = 0.3, shuffle = False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state = 1, test_size = 0.3, shuffle = False)

# print (x1_train.shape)
# print (x1_test.shape)

# 데이터 전처리
# standardscaler

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))

x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))

# print (x_train.shape)
# print (x_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)
# print (x_train_scaled[0, :])
# print (x_train_scaled.shape)
scaler = StandardScaler()
scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)


# ================================전처리 끝================================#

# x_train_scaled = x_train_scaled.reshape(x_train_scaled.shape[0],5, 5)
# x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0], 5, 1)

#print (x_train_scaled.shape)

#DNN 모델 (Dense) 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(25,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2 = Input(shape=(25,))
dense21 = Dense(7)(input2)
dense23 = Dense(4)(dense21)
output2 = Dense(5)(dense23)

from keras.layers.merge import concatenate

from keras.layers import Concatenate

#merge1 = concatenate([output1, output2]) #모델을 사슬처럼 엮다.
merge1 = Concatenate()([output1, output2])
output3 = Dense(1)(merge1)
model = Model(inputs = [input1, input2], outputs = output3)


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, epochs=100, batch_size=1, verbose =1, callbacks=[early_stopping])  

loss, mse = model.evaluate([x1_test_scaled,x2_test_scaled], y1_test, batch_size=1)
print("loss:",loss,"mse:", mse)

y_pred = model.predict([x1_test_scaled,x2_test_scaled])


for i in range(5):
    print('종가:', y1_test[i], '/예측가:', y_pred[i])
