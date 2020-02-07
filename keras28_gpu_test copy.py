import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import datetime

num_samples = 100
height = 71
width = 71
num_classes = 100

start = datetime.datetime.now()
with tf.device('/cpu:0') :
    model = Xception(weights = None, input_shape = (height, width, 3), classes = num_classes)
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop')

    # Generete dummy data
    x = np.random.random((num_samples, height, width, 3))
    y = np.random.random((num_samples, num_classes))

    parallel_model.fit(x, y, epochs = 3, batch_size=16)
    parallel_model.save('./my_model.h5')

end = datetime.datetime.now()
time_delta = end -start
print ("CPU 처리시간 : ", time_delta)