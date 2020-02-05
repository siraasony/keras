from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


# 32 32 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print (x_train.shape)
print (y_train.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print (y_train.shape)
print (x_test.shape)

model = Sequential() 
model.add(Conv2D(7,(2,2),strides=2,padding= 'valid', #7 장, 픽셀 2x2로 짜른다
                 input_shape=(32,32,3))) 
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D(1,1)) 
model.add(Flatten()) # dense 층에 전해주기위해서 4*4*7값을 flatten이 계산
model.add(Dense(10))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=20)

model.fit(x_train, y_train, validation_split=0.2, epochs=100 , batch_size=1, verbose=1, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)

print (acc)