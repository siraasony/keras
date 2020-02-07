#LinearSVC, KNeighborClassifier

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
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x,y)

y_pred= model.predict(x)
print(y_pred)
print(accuracy_score(y, y_pred))
#print(model.score(x,y))




#결과 출력
#print("\n Accuracy: %.4f" % (model.evaluate(x,y)[1]))

# x_test = [[0,0],[1,0],[0,1],[1,1]]
# y_predict = model.predict(x_test)
# print(y_predict)
# print(accuracy_score([0,1,1,0],y_predict))