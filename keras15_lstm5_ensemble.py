from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM , Input


# 1. Data
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20,30,40],[30,40,50],[40,50,60] ])  
y1 = array([4,5,6,7,8, 9, 10, 11, 12, 13, 50, 60, 70])


x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],
           [60,70,80],[70,80,90],[80,90,100], [90,100,110], [100,110,120],
           [2,3,4],[3,4,5],[4,5,6] ])  
y2 = array([40,50,60,70,80, 90, 100, 110, 120, 130, 5, 6, 7])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)





from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_tensor_1 = Input(shape=(3, 1))
hiddenlayers_1 = LSTM(16, activation = 'relu')(input_tensor_1)
output_tensor_1 = Dense(5)(hiddenlayers_1)

input_tensor_2 = Input(shape=(3, 1))
hiddenlayers_2 = LSTM(16, activation = 'relu')(input_tensor_2)
output_tensor_2 = Dense(5)(hiddenlayers_2)

from keras.layers.merge import concatenate , Add 

# merged_model = concatenate([output_tensor_1, output_tensor_2])
merged_model = Add()([output_tensor_1, output_tensor_2])

middle_1 = Dense(4)(merged_model)
middle_2 = Dense(7)(middle_1)
middle_3 = Dense(1)(middle_2)  # merge된 마지막 layer


output_tensor_3 = Dense(8)(middle_3)        # 첫 번째 아웃풋 모델
output_tensor_3 = Dense(1)(output_tensor_3)

output_tensor_4 = Dense(8)(middle_3)        # 두 번째 아웃풋 모델
output_tensor_4 = Dense(1)(output_tensor_4)

model = Model(inputs=[input_tensor_1,input_tensor_2], outputs=[output_tensor_3, output_tensor_4]) 


model.summary()


# 3. 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])  
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')

model.fit([x1, x2], [y1, y2], epochs=1000, batch_size=1, verbose =1, callbacks = [early_stopping])  


# 4. 평가예측
loss= model.evaluate([x1, x2], [y1, y2], batch_size=1)
print(loss) 



x1_input = array([[6.5,7.5,8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])  # (3, ) -> (1,3) -> (1,3,1)
x2_input = array([[6.5,7.5,8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])

x1_input = x1_input.reshape(4,3,1)
x2_input = x2_input.reshape(4,3,1)

y_predict = model.predict([x1_input, x2_input])
print(y_predict)
