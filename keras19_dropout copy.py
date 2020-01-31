from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization


# 1. Data
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20,30,40],[30,40,50],[40,50,60] ])  
y = array([4,5,6,7,8, 9, 10, 11, 12, 13, 50, 60, 70])
#print(x.shape)  # (13, 3)
#print(y.shape)  # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

# 2. model
model = Sequential()
model.add(LSTM(10, input_shape = (3,1)))   # (열, 몇 개씩 자를지)
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(1))
#dropout을 했을 때 성능이 좋아진다는 것은 이론적으로는 맞지만 때에 따라서 다르기 때문에, 해보면서 맞춰가는 것이 좋다.
#특히 dropout이랑 BatchNormalization은 함께 사용하지 않는다. 둘 중에 필요한 것 사용하는 편
#

#activation defalut -> linear / but relu is the best
#return_sequences 해야지 lstm층과 dense와 연결되는 이유: 


model.summary()


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])  
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')
model.fit(x, y, epochs=1000, batch_size=1, verbose =1, callbacks = [early_stopping])  


# 4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae) 

x_input = array([[6.5,7.5,8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])  # (3, ) -> (1,3) -> (1,3,1)
x_input = x_input.reshape(4,3,1)

y_predict = model.predict(x_input)
print(y_predict)
