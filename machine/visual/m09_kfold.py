import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators

warnings.filterwarnings('ignore')

#데이터 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding = 'utf-8')

y= iris_data.loc[:,"Name"]
x= iris_data.loc[:,["SepalLength","SepalWidth","PetalLength",
                    "PetalWidth"]]


#classifier 알고리즘 모두 추출하기 ---1
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter = "classifier")

kfold_cv=KFold(n_splits=10, shuffle=True) #5개씩 자르는

for(name, algorithm) in allAlgorithms:
    #각 알고리즘 객체 생성하기-------2
    model= algorithm()
    
    if hasattr(model, "score"): #model중 score가 가능한 얘들만 쓰겠다 
        scores = cross_val_score(model,x,y, cv=kfold_cv)
        
        print(name, "의 정답률 =")
        print(scores)


        
    # #학습하고 평가하기 ----3
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print(name, "의 정답률 =", accuracy_score(y_test, y_pred))

