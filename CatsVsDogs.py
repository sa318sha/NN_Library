from Library.Models.Sequential import Sequential
from Library.Optimizers.Optimizer import Optimizer
from Library.metrics.Metrics import metrics
from Library.Optimizers.Adam import Adam
from Library.Layers.Layer_Dense import Layer_Dense
from Library.Layers.Convolutional_Layer import Convolutional_Layer
from Library.Layers.MaxPool2D import MaxPool2D
from Library.Layers.Flatten import Flatten
from Image_Manipulation.image_data_generator import ImageDataGenerator
import numpy as np
import pandas as pd

np.random.seed(0)

learning_rate =0.0001
train_directory = 'Data/processed_images/train'
print('train directory', train_directory)
train_batches = ImageDataGenerator(None).flow_from_directory(target_size = (224,224),classes=['cat','dog'], shuffle = True, directory =train_directory)

data = train_batches[0]
# print('train batches',train_batches[1])
target = np.argmax(train_batches[1], axis=1)
# print('target', target)
spliced_data = data[:15]
spliced_target = target[:15]


print('length of data',len(spliced_target))

# print(type(spliced_data), type(spliced_target))
CatsvsDogsModel = Sequential([

  Convolutional_Layer(1,8,(3,3), optimizer = Adam(learning_rate)),
  MaxPool2D((3,3),2),
  Convolutional_Layer(8,32,(3,3), optimizer = Adam(learning_rate)),
  MaxPool2D((3,3),3),
  Flatten(),
  Layer_Dense(46208,100, optimizer = Adam(learning_rate)),
  Layer_Dense(100,10, optimizer = Adam(learning_rate)),
  Layer_Dense(10,2, activation = 'softmax', optimizer = Adam(learning_rate))

])

CatsvsDogsModel.compile(loss_function= metrics.categorical_crossEntropy, metrics=[metrics.accuracy])
print('compiled')
CatsvsDogsModel.fit(10,20,spliced_data,spliced_target,validation_split=0.25)
print('fitting')