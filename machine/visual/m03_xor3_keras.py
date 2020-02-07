from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#데이터
x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
y_train = np.array([0,1,1,0])

#print(x_train.shape)
x_train = x_train.reshape(4, 2)

#모델
#model = LinearSVC()
#model = KNeighborsClassifier(n_neighbors=1)
model = Sequential()

model.add(Dense(32, input_shape =(2,))) 
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#model.summary()

#훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,epochs = 10, batch_size = 1)

#평가예측
x_test = np.array([[0,0],[1,0],[0,1],[1,1]])
x_test = x_test.reshape(4, 2)
y_test = np.array([0,1,1,0])


y_predict = model.predict(x_test)
print(y_predict)

loss, acc = model.evaluate(x_test, y_test,batch_size=1)
print('accuracy: ',acc)
