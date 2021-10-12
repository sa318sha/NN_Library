

import numpy as np

class Convolutional_Layer():
  def __init__(self,numberOfFilters,filterWidth,filterHeight,padding=True,activation = 'relu'):

    self.width = filterWidth
    self.height = filterHeight
    self.numberOfFilters = numberOfFilters
    self.padding = padding
    self.activation = activation
    self.weights = np.random.randn(self.numberOfFilters, self.height,self.width) # randn(i,j) creates random shape of rows i and columns j with normal distrubution where i is the number of neurons in current layer and j is number of neurons in previous layer or input
    # we are diubg inputs first and then neurons second because, although it should be the other way around, this way we dont need to transpose when doing the dot product so saves computational speed

    self.biases = np.zeros((1,numberOfFilters))

    
    self.weightChanges = 0
    self.biasChanges = 0
    self.V_dW = np.zeros(shape=(self.weights.shape))
    print('VdW initialized')
    self.V_dB = np.zeros(shape=(self.biases.shape))
    self.S_dW = np.zeros(shape=(self.weights.shape))
    self.S_dB = np.zeros(shape=(self.biases.shape))
    self.t = 0
  
  def set_weights(self,newWeights):

    self.weights = newWeights

  def set_bias(self,newBias):

    self.biases = newBias

  def set_values(self,newBias,newWeights):

    self.weights = newWeights
    self.biases = newBias

  def layer_update(self,batch_size,optimizer):

    if(optimizer.name == 'Adam'):
      # print('weights b4', self.weights)
      # print('before',self.weights)

      self.V_dW,self.V_dB, self.S_dW, self.S_dB, self.t = optimizer.update(self.V_dW, self.V_dB,self.S_dW, self.S_dB, self.weightChanges,self.biasChanges,batch_size,self.weights,self.biases,self.t)
      # print("T IS", self.t)
      # print('after',self.weights)
      # print('V_dw after optimizer in this layer', self.V_dW)
      # print('after function finish S_dw',self.S_dW)
      # print('weights after', self.weights)
      # print('weights shape', self.weights.shape)


    elif(optimizer.name == 'Gradient_Descent'):
      self.weights, self.biases = optimizer.update(self.weights,self.biases,self.weightChanges,self.biasChanges,batch_size)
    
    else:
      print('no optimizer used')

  def backPropogation():

    
    pass


  def forward(self,image):

    if self.padding == True:
      pass #create a function that padds according tot he filter size
      #however until this is created we are going to assume that the padding is same and filter size is 3x3
    padded_img = np.zeros((image.shape[0], image.shape[1]+2, image.shape[2]+2))

    for img in range(image.shape[0]):
      padded_img[img] = np.r_[np.zeros((1,image.shape[2]+2)),
                        np.c_[np.zeros((image.shape[1],1)),
                        image[img],
                        np.zeros((image.shape[1],1))],
                        np.zeros((1,image.shape[2]+2))]



    
    width = padded_img.shape[2]-self.weights.shape[2]+1
    height = padded_img.shape[1]-self.weights.shape[1]+1



    # might need more complex width and height calculations
    filterAppliedImages = np.zeros((self.weights.shape[0],height, width))
    for i in range(padded_img.shape[0]):

      for z in range(self.weights.shape[0]):

        for x in range(width):
          # print('x',x)
          for y in range(height):
            # print('z,y,x',z,y,x)
            # print('y',y)
            # print('image at x and y respectivly', x,y)
            # print(image[x:x+width+1, y: y+height+1])

            filterAppliedImages[i,z,x,y] = np.sum(image[i,y:y+self.weights.shape[1],x:x+self.weights.shape[2]]*self.weights[z])
            # newImage[x,y] = np.sum
      
    self.zValues = filterAppliedImages + self.biases

# print(Cnn1.weights[0])


# print('newImage', newImage)

    #image is a 2d array
    #not sure which activation functions are allowed in convolutional layers
    if self.activation == 'relu': 

      self.output = np.maximum(0,self.zValues)
      #output will be in teh shape of hoepfully [number of images, number of filters, height, width]

    # not 
    # elif self.activation == 'softmax':
    #   exp_values = np.exp(self.zValues -np.max(self.zValues,axis=1,keepdims=True))
    #   norm_values = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
    #   self.output = norm_values

    elif self.activation == 'none':
      pass
    else:
      print('poo')

