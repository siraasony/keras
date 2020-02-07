from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

x_train = [[0,0],[1,0],[0,1],[1,1]]
y_train = [0,0,0,1]

model = LinearSVC()

model.fit(x_train,y_train)

x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)
print(y_predict)
print(accuracy_score([0,0,0,1],y_predict))