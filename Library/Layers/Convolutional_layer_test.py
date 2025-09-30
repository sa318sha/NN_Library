import numpy as np
from Convolutional_Layer import Convolutional_Layer
from Flatten import Flatten
from MaxPool2D import MaxPool2D
import sys,os

print('sys path',sys.path)



# from Utilities.Timing.Timer_decorator import non_return_timer
np.random.seed(0)

batch = 1
chanels = 1
height = 10
width = 10


image = np.random.rand(batch,1,height,width)


filterHeight = 3
filterWidth = 3

Layer = Convolutional_Layer( in_channels=1,out_channels=2,filterDimensions = (filterHeight,filterWidth))
Layer2 = Convolutional_Layer(in_channels=2,out_channels=8,filterDimensions = (filterHeight,filterWidth))
pool = MaxPool2D((2,2),2)
pool2 = MaxPool2D((2,2),2)
flattenLayer = Flatten()


Layer.forward(image)
print('Convo2d Layer 1 output',Layer.output, Layer.output.shape)
pool.forward(Layer.output)
print('pool Layer 1 output',pool.output, pool.output.shape)

Layer2.forward(pool.output)
print('Convo2d Layer 1 output',Layer2.output, Layer2.output.shape)
pool2.forward(Layer2.output)
print('pool Layer 1 output',pool2.output, pool2.output.shape)

flattenLayer.forward(pool2.output)
print('flatten Layer output',flattenLayer.output, flattenLayer.output.shape)

print('end of forward')

delta = np.random.rand(1,72)

print('delta',delta,delta.shape)

print('starting backprop')

delta = flattenLayer.backPropogation(delta)
print('delta after flatten backprop',delta,delta.shape)
delta = pool2.backPropogation(delta)
print('delta after pool',delta,delta.shape)

# delta = Layer2.backPropogation(delta)
# print('delta after layer2 backprop',delta,delta.shape)
# delta = pool.backPropogation(delta)
# print('delta after pool',delta,delta.shape)
delta = Layer.backPropogation(delta)
print('delta after layer1 backprop',delta,delta.shape)




# image = np.random.randn(3,3,3)

# 1/2/3/3 is one image that has two filters 3 
#image is in shape of Number of images channels Height,width


# print('image',image)


# print('layer filters',Layer.weights)

# Layer.forward(image)
# outputImage = Layer.output


# print('filtered images', outputImage)
# print('image shape', image.shape)
# print('filtered images shape', outputImage.shape)

# flattenLayer.forward(outputImage)

# delta = np.random.randn(image.shape[0],2*height*width)
# print('delta and delta shape',delta,delta.shape)
# delta = flattenLayer.backPropogation(delta)
# print('delta 1st backprop',delta)
# print('delta shape', delta.shape)
# delta = Layer.backPropogation(delta)

# print('layer after,', Layer.zValues)
# print('output:', Layer.output)
# print('biasis', Layer.biases)

# zValues = Layer.zValues


# test1 = np.arange(-1,8).reshape(1,1,3,3)

# print(test1)
# print(np.maximum(zValues,0))

