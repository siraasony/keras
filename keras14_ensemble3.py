#1. 데이터
import numpy as np

x1= np.array([range(1,101),range(101,201),range(301,401)])
#x2= np.array([range(1001,1101),range(1101,1201),range(1301,1401)])

#x2= np.array([range(1,101)])

y1 = np.array([range(1,101), range(101,201), range(301,401)]) 
y2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)]) 
y3 = np.array([range(1,101), range(101,201), range(301,401)]) 

# print(x1.shape) # (3,100)
# print(y1.shape)  # (1,100)
#print(y2.shape)  # (1,1)

# x=x.reshape(100,3)
# y=y.reshape(100,1)
# y2=y2.reshape(1,1)

x1= np.transpose(x1) 
#x2= np.transpose(x2)
y1= np.transpose(y1)
y2= np.transpose(y2)
y3= np.transpose(y3)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, 
                            test_size = 0.2,  shuffle = False) #일단 Train과 Test로 나눠 준 다음

x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test,
                            test_size = 0.5, shuffle = False) # Test에서 val&test로 나눠준다

y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3,
                            test_size = 0.2, shuffle = False) #일단 Train과 Test로 나눠 준 다음

y2_val, y2_test, y3_val, y3_test = train_test_split(y2_test, y3_test,
                            test_size = 0.5, shuffle = False) # Test에서 val&test로 나눠준다

#원칙적으로는 원본데이터에서 학습데이터 테스트 데이터를 분리해줘야 한다.


print(y1_val)
print(y2_train)   
print(y3_test)

# print(x.shape) 
# print(y.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense3 = Dense(3)(dense1)
output1 = Dense(4)(dense3)

# input2 = Input(shape=(3,))
# dense21 = Dense(7)(input2)
# dense22 = Dense(4)(dense21)
# output2 = Dense(5)(dense22)


# from keras.layers.merge import concatenate
# merge1 = concatenate([output1, output2]) #output1과 2를 합치다

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(1)(middle2) #현재 merge된 마지막 레이어

output_1 = Dense(2)(output1) #1번째 아웃풋 모델
output_1 = Dense(3)(output_1)

output_2 = Dense(4)(output1) #2번째 아웃풋 모델
output_2 = Dense(4)(output_2)
output_2 = Dense(3)(output_2)

output_3 = Dense(5)(output1) #3번째 아웃풋 모델
output_3 = Dense(3)(output_3)

model = Model(inputs = input1,
               outputs = [output_1, output_2, output_3])
model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train, y3_train] , epochs =100, batch_size=10,
           validation_data= (x1_val, [y1_val, y2_val, y3_val]))

# #4. 평가
aaa = model.evaluate(x1_test, [y1_test,y2_test, y3_test], batch_size=1) #아웃풋 3개인 모델, 7개 (전체로스, 1,2,3번째 모델로스, mae1,2,3)

print(aaa)
# #1. 변수를 1개
# #2. 변수를 mse갯수별로

# print('mse: ', mse)

x_prd1 = [[201,202,203],[204,205,206],[207,208,209]]
#x_prd2 = [[201,202,203],[204,205,206],[207,208,209]]

x_prd1= np.transpose(x_prd1)
#x_prd2= np.transpose(x_prd2)

aaa = model.predict(x_prd1, batch_size=10) 
print(aaa)

y1_predict= model.predict(x1_test, batch_size=10)
print(y1_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))
#print("RMSE:", RMSE(y1_test, y1_predict))

rmse1 = RMSE(y1_predict[0], y1_test)
rmse2 = RMSE(y1_predict[1], y2_test)
rmse3 = RMSE(y1_predict[2], y3_test)

rmse = (rmse1 +rmse2 + rmse3) /3
print("RMSE : ", rmse)

#R2 구하기
from sklearn.metrics import r2_score

r2_y_predict1 = r2_score(y1_test, y1_predict[0])
r2_y_predict2 = r2_score(y2_test, y1_predict[1])
r2_y_predict3 = r2_score(y3_test, y1_predict[2])
r2_y_predict = (r2_y_predict1+r2_y_predict2+r2_y_predict3) / 3

print("R2:", r2_y_predict)
