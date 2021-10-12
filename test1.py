from numpy.core.fromnumeric import shape
from numpy.core.numeric import NaN
from Library.Layers.Convolutional_Layer import Convolutional_Layer
import numpy as np
np.random.seed(0)
Cnn1 = Convolutional_Layer(2,3,3)
image = np.random.rand(2,3,3)
padded_img = np.zeros((image.shape[0], image.shape[1]+2, image.shape[2]+2))

for img in range(image.shape[0]):
  padded_img[img] = np.r_[np.zeros((1,image.shape[2]+2)),
                  np.c_[np.zeros((image.shape[1],1)),
                  image[img],
                  np.zeros((image.shape[1],1))],
                  np.zeros((1,image.shape[2]+2))]

print('image',image)
print('padded image',padded_img)
print('weights/filter', Cnn1.weights)


print('image shape', padded_img.shape)
print('image width',padded_img.shape[2] )
print('image height', padded_img.shape[1])
print('cnn shape', Cnn1.weights.shape)
print('cnn width', Cnn1.weights.shape[2])
width = padded_img.shape[2]-Cnn1.weights.shape[2]+1
height = padded_img.shape[1]-Cnn1.weights.shape[1]+1
print('processed width and height', width, height)

newImage = np.zeros((padded_img.shape[0],Cnn1.weights.shape[0],width, height))
#new images shape will be #number of images, #number of filters, width and height
# print('new image',newImage)
print('new image dimensions', newImage.shape)
print('img', padded_img.shape[0])
for i in range(padded_img.shape[0]):
  
  for z in range(Cnn1.weights.shape[0]):

    for x in range(width):
      # print('x',x)
      for y in range(height):
        # print('z,y,x',z,y,x)
        # print('y',y)
        # print('image at x and y respectivly', x,y)
        # print(image[x:x+width+1, y: y+height+1])

        newImage[i,z,x,y] = np.sum(
                                padded_img[
                                  i,
                                  y:y+Cnn1.weights.shape[1],
                                  x:x+Cnn1.weights.shape[2]]*Cnn1.weights[z])
        # newImage[x,y] = np.sum
        
# print(Cnn1.weights[0])


print('newImage', newImage)
# multiplied = np.multiply(image, x2)

# # print(multiplied)

# print(Cnn1.biases)

# x = np.arange(4).reshape(2,2)
# y = np.arange(2,6).reshape(2,2)
# print(x)
# print(y)
# # new_array_dot = np.dot(x,y)
# new_array = x*y
# sum = np.sum(new_array)
# print(new_array)
# print(sum)
# print(new_array_dot)
# x = np.arange(4)
# xx = x.reshape(4,1)
# y = np.ones(5)
# print(x)
# print(xx)
# print(y)
# print(xx*y)