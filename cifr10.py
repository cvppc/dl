import keras as keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train/255.0
x_train.shape
x_test=x_test/255.0
x_test.shape
plt.imshow(x_test[215])
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=3,padding="same", activation="relu", input_shape=[32,32,3]))
model.add(Conv2D(filters=32,kernel_size=3,padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2,strides=2,padding='valid'))
model.add(Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2,strides=2,padding='valid'))
model.add(Flatten())
model.add(Dropout(0.5,noise_shape=None,seed=None))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()

# Adam
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
model.fit(x_train,y_train,epochs=15)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

# RMS Prop
model.compile(loss="sparse_categorical_crossentropy",optimizer="RMSprop", metrics=["sparse_categorical_accuracy"])
model.fit(x_train,y_train,epochs=15)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

# Adaptive Gradient Descent
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adagrad", metrics=["sparse_categorical_accuracy"])
model.fit(x_train,y_train,epochs=15)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Stochastic Gradient Descent
model.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=["sparse_categorical_accuracy"])
model.fit(x_train,y_train,epochs=15)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

