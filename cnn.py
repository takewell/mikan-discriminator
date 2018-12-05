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
    label_list.append(label)
    image_path = dir + "/" + filename
    image = np.array(Image.open(image_path).convert("L").resize((244,244)))
    image_list.append(image)

image_list = np.array(image_list)
label_list = np.array(label_list)
label_list = np_utils.to_categorical(label_list)

(train_data, test_data, train_label, test_label) = train_test_split(image_list, label_list, test_size=0.3, random_state=111)
train_data = train_data.reshape(-1, 244, 244, 3)
test_data = test_data.reshape(-1, 244, 244, 3)

input_shape = train_data[0].shape

print(input_shape)