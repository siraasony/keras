from numpy import array
                # 10, 4
def split_sequence(sequence, n_steps): #n_steps: 몇개씩 자를건가
    x, y = list(), list()
    for i in range(len(sequence)):       # 10
        end_ix = i + n_steps             # 0 + 4 = 4
        if end_ix > len(sequence)-1 :    # 4 > 10 -1
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #  seq_x = 0,1,2,3 / seq_y = 4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_steps = 3
x, y = split_sequence(dataset, n_steps) # 7행 3열짜리 행렬과 7컴마짜리 벡터가 나왔음.
print (x)
print (y)
x = x.reshape(x.shape[0], x.shape[1], 1)
print (x.shpae)

# for i in range(len(x)):
#     print(x[i], y[i])
  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle = True)
    
# 실습 : DNN 모델 (Dense) 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1), return_sequences = True))   # (열, 몇 개씩 자를지)
model.add(LSTM(5, activation = 'relu', return_sequences = True))
model.add(LSTM(6, activation = 'relu', return_sequences = True))
model.add(LSTM(7, activation = 'relu', return_sequences = True))
model.add(LSTM(8, activation = 'relu', return_sequences = True))
model.add(LSTM(9, activation = 'relu', return_sequences = True))
model.add(LSTM(10, activation = 'relu', return_sequences = True))
model.add(LSTM(1, activation = 'relu', return_sequences = False))
model.summary()

# loss, 출력
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x, y, epochs=100, batch_size=1, verbose =1)  
loss, mse = model.evaluate(x, y, batch_size=1)
print("loss:",loss) 

# 90, 100, 110을 예측
x_input = array([90,100,110])
x_input = x_input.reshape(1,3,1)
y_predict1 = model.predict(x_input)
print (y_predict1)


'''
0, 1, 2, 3 / 4
1, 2, 3, 4 / 5

5, 6, 7, 8 / 9
'''