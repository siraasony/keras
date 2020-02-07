#keras로 변경하기
import numpy as np 
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

#데이터 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding = 'utf-8',
                        names=['a','b','c','d','y'])


#붓꽃데이터를 레이블과 입력 데이터로 분리하기
y= iris_data.loc[:,"y"]
x= iris_data.loc[:,["a","b","c","d"]]

y=y.replace(['Iris-virginica','Iris-setosa','Iris-versicolor'
            ],[0,1,2])
# print(y)

(x_train, x_test, y_train, y_test) = train_test_split(
    x,y, train_size=0.8, random_state=1)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x.shape) #(150,4)
#모델의 설정
model = Sequential()
model.add(Dense(12,input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()
#모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 모델 실행
model.fit(x_train,y_train,epochs=10, batch_size=1)


#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x_test,y_test)[1]))

