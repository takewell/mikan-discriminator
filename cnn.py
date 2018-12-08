import keras
from keras import backend
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
import os

image_list = []
label_list = []

data_path = "data"

for label in os.listdir(data_path):
  dir = data_path + "/" + label
  for filename in os.listdir(dir):
    if label == "failure":
      label_list.append("0")
    else:
      label_list.append("1")

    image_path = dir + "/" + filename
    width, height = 1280, 960
    crop_args = (240, 80, width - 240, height - 80)
    image = np.array(Image.open(image_path).crop(crop_args).resize((244,244)))
    image_list.append(image)

image_list = np.array(image_list)
label_list = np.array(label_list)
label_list = np_utils.to_categorical(label_list)

(train_data, test_data, train_label, test_label) = train_test_split(image_list, label_list, test_size=0.3, random_state=111)
train_data = train_data.reshape(-1, 244, 244, 3)
test_data = test_data.reshape(-1, 244, 244, 3)

input_shape = train_data[0].shape
batch_size = 64
epochs = 5
kernel_size = (4,4)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=kernel_size, input_shape=input_shape, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=kernel_size, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=kernel_size, activation="relu"))
model.add(Conv2D(filters=64, kernel_size=kernel_size, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs)

scores = model.evaluate(test_data, test_label, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])