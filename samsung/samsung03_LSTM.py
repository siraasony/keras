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
    
x, y = split_xy5(samsung, 5, 1)
print (x.shape)
print (y.shape)
print (x[0,:], '\n', y[0])

# ================================묶음으로 잘라내기================================#

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.3, shuffle = False)

print (x_train.shape)
print (x_test.shape)

# 데이터 전처리
# standardscaler

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print (x_train.shape)
print (x_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print (x_train_scaled[0, :])
print (x_train_scaled.shape)


# ================================전처리 끝================================#

x_train_scaled = x_train_scaled.reshape(x_train_scaled.shape[0],5, 5)
x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0],5, 5)

print (x_train_scaled.shape)

#DNN 모델 (Dense) 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, activation = 'relu', input_shape = (5,5)))   # (열, 몇 개씩 자를지)
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))


model.summary()

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=1, verbose =1, callbacks=[early_stopping])  

loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print("loss:",loss,"mse:", mse)

y_pred = model.predict(x_test_scaled)


for i in range(5):
    print('종가:', y_test[i], '/예측가:', y_pred[i])
