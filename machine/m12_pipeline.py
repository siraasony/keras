import pandas as pd


iris_data = pd.read_csv("./data/iris2.csv", encoding = 'utf-8')

x =  iris_data.loc[:'Name']
y =  iris_data.loc[:'SepalLength', 'Sepalwidth', 'PetalLength', 'PetalWidth']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size = 0.7, suffle = True)

from sklearn.pipeline import Pipeline
from Sklearn.preprocessing import MinMaxScaler

pipe = pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])

model.fit(x_train, y_train)

print ('test score', model.score(x_test, x_test))