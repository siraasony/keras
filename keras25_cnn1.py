from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential() 
model.add(Conv2D(7,(2,2),strides=2,padding= 'valid', #7 장, 픽셀 2x2로 짜른다
                 input_shape=(5,5,1))) 
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D(1,1)) 
model.add(Flatten()) # dense 층에 전해주기위해서 4*4*7값을 flatten이 계산
model.add(Dense(1))

model.summary()


