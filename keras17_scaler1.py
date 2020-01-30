from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

# 1. Data
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000],[30000,40000,50000],[40000,50000,60000],[100,200,300]])  
y = array([4,5,6,7,8, 9, 10, 11, 12, 13, 500000, 60000, 70000, 400])
print(x.shape)  # (14, 3)
print(y.shape)  # (14,)

#----------------------------------------------------------------실습
#train은 10개, 나머지는 테스트 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle = True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#fit -> training 할 때
#scaler.fit(x) # fit -> transform
#x = scaler.transform(x) #값이 0-1사이로 나온다. 모든 데이터 값을 표준화 시킨다.
#print (x1) 
scaler = StandardScaler()
#fit -> training 할 때
scaler.fit(x_train) # fit -> transform
x_train = scaler.transform(x_train) #값이 0-1사이로 나온다. 모든 데이터 값을 표준화 시킨다.

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print (x_train)
print (x_test)

#Dense model 구현
model = Sequential()
model.add(Dense(5, input_dim =3))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

# R2지표 잡기
early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x, y, epochs=100, batch_size=1, verbose =1, callbacks = [early_stopping])  

loss, mse = model.evaluate(x, y, batch_size=1)
print("loss:",loss,"mse:", mse) 

from sklearn.metrics import r2_score
y_predict = model.predict(x_test, batch_size = 1)
r2_y_predict = r2_score(y_test, y_predict)
print ("R2:", r2_y_predict)

x_input = array([250,260,270])
x_input = x_input.reshape(1,3)
y_predict1 = model.predict(x_input)
print (y_predict1)

#결과가 구린 이유: x 값은 전처리가 되어있고 y 값은 전처리가 되어있지 않아서
#그러나 쌍은 변하지 않음. 그래서 y는 전처리를 할 필요가 없음.
#학습할 때 train/test split으로 분리해서 사용함.



