import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np 
from keras.layers import Dense
from keras.models import Sequential


#1 데이터 읽어들이기
wine = pd.read_csv('./data/winequality-white.csv', sep = ';', encoding = 'utf-8')

#2 데이터를 레이블과 입력데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# print(x.shape)
# print(y.shape)

#모델의 설정
model = Sequential()
model.add(Dense(12,input_shape=(11,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

#모델 컴파일
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(x,y,epochs=100, batch_size=10)

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x_test,y_test)[1]))