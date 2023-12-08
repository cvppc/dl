from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train.shape
x_valid.shape
x_train.dtype
x_train.min()
x_train.max()
x_train[0]

image = x_train[0]
plt.imshow(image, cmap='gray')

y_train[0]
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)
x_train.shape
x_train[0]
x_train = x_train / 255
x_valid = x_valid / 255
x_train.dtype
x_train.min()
x_train.max()

num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
25
y_valid = keras.utils.to_categorical(y_valid, num_categories)
y_train[0:9]

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid))