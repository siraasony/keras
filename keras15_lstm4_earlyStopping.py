from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM


# 1. Data
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20,30,40],[30,40,50],[40,50,60] ])  
y = array([4,5,6,7,8, 9, 10, 11, 12, 13, 50, 60, 70])
print(x.shape)  # (13, 3)
print(y.shape)  # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)



# 2. model
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1)))   # (열, 몇 개씩 자를지)
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])  
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')
# monitor : 지켜볼 항목 / val_loss, val_acc, loss, acc
# min_delta : 개선되고 있다고 판단하기 위한 최소 변화량 / 만약 변화량이 min_delta 보다 적은 경우에는 개선이 없다고 판단
# patience : 개선이 없다고 바로 종료하지 않고 개선이 없는 에포크를 얼마나 기다려줄지/ 만약 10이라고 지정하면 개선이 없는 에포크가 10번째 지속될 경우 학습 종료
# mode : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준 설정 / auto, min, max
model.fit(x, y, epochs=1000, batch_size=1, verbose =1, callbacks = [early_stopping])  


# 4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae) 



x_input = array([[6.5,7.5,8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])  # (3, ) -> (1,3) -> (1,3,1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
