import numpy as np
from numpy import random
from numpy.lib.arraypad import pad
np.random.seed(0)

image = np.random.randn(1,1,4,4)
weight = np.random.randn(2,3,3)

print('weights',weight)

print('Avalues',image)

zValues = np.zeros(image.shape)
print(zValues)

padded_img = np.zeros((image.shape[0],image.shape[1], image.shape[2]+2, image.shape[3]+2))


for img in range(image.shape[0]):
    for channel in range(image.shape[1]):
        padded_img[img,channel] = np.r_[np.zeros((1,image.shape[3]+2)),
                                  np.c_[np.zeros((image.shape[2],1)),
                                  image[img,channel],
                                  np.zeros((image.shape[2],1))],
                                  np.zeros((1,image.shape[3]+2))]


filterWidth = 3
filterHeight = 3

changeInWeights = np.zeros(weight.shape)

print(changeInWeights)


delta = np.random.randn()

for filter in range(changeInWeights.shape[0]):
  for row in range(changeInWeights.shape[1]):
    for column in range(changeInWeights.shape[2]):
      changeInWeights[filter,row,column] = np.sum(padded_img[0,
                                                            0,
                                                            row : padded_img.shape[2] - filterHeight + 1 + row,
                                                            column : padded_img.shape[3] - filterWidth + 1 + column])

print(changeInWeights)
      # print('change in weights',changeInWeights[filter,row,column])
      # print('area of sum', padded_img[0,0,row:changeInWeights.shape[0]-filterHeight+1+row,column:column+filterWidth-1])
      # print('total row and height')
      # changeInWeights[filter,row,column] = 
# print(changeInWeights)