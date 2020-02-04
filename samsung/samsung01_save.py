import numpy as np
import pandas as pd

df1 = pd.read_csv("./data/samsung.csv", index_col=0, header=0, encoding='cp949', sep=',') # 날짜 빠지고 나머지 데이터만 남는다. 
print (df1)
print (df1.shape)

df2 = pd.read_csv("./data/kospi200.csv", index_col=0, header=0, encoding='cp949', sep=',')
print (df2)
print (df2.shape)

# 삼성전자의 모든 데이터
# 숫자 사이에 ,를 제거하기 위해서 파라미터 thousand를 활용하는 방법도 있다.
# df1 = df1.replace('\D', '', regex=True).astype(int)
for i in range(len(df1.index)) :
    for j in range(len(df1.iloc[i])) :
        df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))
    
# kospi200의 거래량
for i in range(len(df2.index)) :
    df2.iloc[i,4] = int(df2.iloc[i,4].replace(',', ''))

print (df1)
print (df1.shape)

# 내림차수로 되어있는 데이터를 오름차순으로 변경해준다.
df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print (df2)

# pandas to numpy why? 계산 속도가 빠르다.
df1 = df1.values
df2 = df2.values
print (type(df1), type(df2))
print (df1.shape, df2.shape)

np.save('./data/samsung.npy', arr = df1)
np.save('./data/kospi200.npy', arr = df2)

# ================================데이터 로드 끝================================#

