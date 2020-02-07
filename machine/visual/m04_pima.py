from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

seed=0
numpy.random.seed(seed) # numpy 나 tensorflow에 random하겟냐 말겟냐
tf.set_random_seed(seed)

#데이터로드
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=',')
x=dataset[:,0:8]
#print(x.shape)
y=dataset[:,8]
#print(y.shape)
#모델의 설정
model = Sequential()
model.add(Dense(12,input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 모델 실행
model.fit(x,y,epochs=200, batch_size=10)

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(x,y)[1]))

