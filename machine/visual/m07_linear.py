from sklearn.datasets import load_boston
boston = load_boston()

x= boston.data
y= boston.target

print(x.shape)
print(y.shape)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

ridge=Ridge()
ridge.fit(x,y)

lr=LinearRegression()
lr.fit(x,y)

lasso= Lasso()
lasso.fit(x,y)

print(ridge.score(x,y))
print(lr.score(x,y))
print(lasso.score(x,y))

# x_test = [[0,0],[1,0],[0,1],[1,1]]
# y_predict = model.predict(x_test)
# print(y_predict)
# print(accuracy_score([0,0,0,1],y_predict))
