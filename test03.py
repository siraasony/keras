import numpy as np
import tensorflow as tf
from keras import models
from keras import layers


x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

model = models.Sequential()

model.add(layers.Dense(256, input_dim = 1))
model.add(layers.Dense(256))
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_data, y_data, epochs=50, batch_size=1)

x_prd = np.array([11, 12, 13])
print(model.predict(x_prd))
