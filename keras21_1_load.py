#1. 데이터
import numpy as np

x = np.array([range(1,101), range(101,201), range(301,401)]) 
y = np.array([range(101,201)])
# y2 = np.array(range(101,201))

print(x.shape) #(3,100)
print(y.shape) #(1,100)
# print(y2.shape) #(100, )

#1.1 데이터 차원변경(shape)
x = x.reshape(100,3)
y = y.reshape(100,1)


#1.2 데이터 차원변경(transpose)

# x = np.transpose(x)
# y = np.transpose(y)

print(x.shape)
print(y.shape)

#1-1. 데이터나누기(사이킷런을 사용해서)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = False) #일단 Train과 Test로 나눠 준 다음
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, shuffle = False) # Test에서 val&test로 나눠준다
#원칙적으로는 원본데이터에서 학습데이터 테스트 데이터를 분리해줘야 한다.


#2.모델구성
from keras.layers import Dense
from keras.models import load_model
model = load_model("./save/savetest01.h5")
model.add(Dense(10, name= 'dense_name1'))
model.add(Dense(5, name= 'dense_name2'))
model.add(Dense(1, name= 'output_name1'))

model.summary()
#3.훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) 

from keras.callbacks import EarlyStopping, TensorBoard
tb = TensorBoard(log_dir='./graph')

early_stopping = EarlyStopping(monitor = 'loss', patience = 150, mode = 'auto')
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data = (x_val, y_val), callbacks = [early_stopping,tb]) 


#4.평가예측

loss, mse = model.evaluate(x_test,y_test,batch_size=1)
print('mse:', mse)

x_prd = np.array([[201,202,203],[204,205,206],[207,208,209]])
x_prd = np.transpose(x_prd) #x_prd가 2행 3열이기때문에 다시 3행 1열로 바꿔주어야 된다.
test = model.predict(x_prd, batch_size=1)
print(test)

y_predict = model.predict(x_test, batch_size = 1) # <——— 이거 중요~! x_test와 y_test 매치가 중요하다

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

print('RMSE : ', RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)