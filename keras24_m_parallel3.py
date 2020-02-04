from numpy import array, hstack
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import TensorBoard


def split_sequence3(sequence, n_steps): #n_steps: 몇개씩 자를건가
    x, y = list(), list()
    for i in range(len(sequence)):      
        end_ix = i + n_steps             
        if end_ix > len(sequence)-1 :   
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :] #  seq_x = 0,1,2,3 / seq_y = 4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([in_seq1[i] +  in_seq2[i] for i in range(len(in_seq1))])

# print (in_seq1.shape) # (10, )
# print (in_seq2.shape) # (10, )
# print (out_seq.shape) # (10, )

in_seq1 = in_seq1.reshape(len(in_seq1), 1) # 백터 3개를 합쳐야해서 reshape을 해준다. 2차원 형태의 변경 필요.
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

# print (in_seq1.shape) # (10, 1)
# print (in_seq2.shape) # (10, 1)
# print (out_seq.shape) # (10, 1) -> (10, 3)의 데이터로 변경하기 위해서 변경해줌.

dataset = hstack((in_seq1, in_seq2, out_seq)) # hstack을 해주면 (10,1) 3개가 쌓여서 (10,3)으로 됨
#print (dataset)

n_steps = 3
x, y = split_sequence3(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print (x)
print (x.shape) # (7, 3, 3)
print (y.shape) # (7, 3)

# 1.함수의 구조 파악
# 2.모델 생성 (DNN -2차원의 입력) - 입력 shape이 다르다.
#x = x.reshape(7,3,3) 

#y = y.reshape((1,24))
print (x.shape)
print (x)

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,3), return_sequences = True))   # (열, 몇 개씩 자를지)
model.add(LSTM(5, activation = 'relu', return_sequences = True))
model.add(LSTM(6, activation = 'relu', return_sequences = True))
model.add(LSTM(7, activation = 'relu', return_sequences = True))
model.add(LSTM(8, activation = 'relu', return_sequences = True))
model.add(LSTM(9, activation = 'relu', return_sequences = True))
model.add(LSTM(10, activation = 'relu', return_sequences = True))
model.add(LSTM(3, activation = 'relu', return_sequences = False))


# 3. 지표는 loss
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(x, y, epochs=100, batch_size=1, verbose =1)  
loss, mse = model.evaluate(x, y, batch_size=1)
print("loss:",loss,"mse:", mse) 

# 4. [[90, 95], [100, 105], [100, 115]] -> 215 predict
x_input = array([[90, 95, 100], [100, 105, 110], [110, 115, 120]])
x_input = x_input.reshape(1,3,3)
y_predict1 = model.predict(x_input)
print (y_predict1)

#회귀모델에서는 accuracy를 사용하지 않는다.