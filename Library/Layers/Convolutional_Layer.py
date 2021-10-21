

import numpy as np

class Convolutional_Layer():
  def __init__(self,in_channels, out_channels,filterDimensions,stride = 1,padding=True,use_bias = False,activation = None):
    
    self.stride = stride
    self.use_bias = use_bias
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filterHeight,self.filterWidth = filterDimensions
    self.numberOfFilters = int(out_channels/in_channels)
    # print('width and height', self.width,self.height)
    print('number of filters', self.numberOfFilters)
    self.padding = padding
    self.activation = activation
    self.weights = np.random.randn(self.numberOfFilters, self.filterHeight,self.filterWidth) # randn(i,j) creates random shape of rows i and columns j with normal distrubution where i is the number of neurons in current layer and j is number of neurons in previous layer or input
    # we are diubg inputs first and then neurons second because, although it should be the other way around, this way we dont need to transpose when doing the dot product so saves computational speed
    # print('weights',self.weights)

    self.biases = np.zeros((1,self.out_channels))

    
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

  def backPropogation(self,delta):
    
    delta = self.create_padding(delta)
    # print('padded delta',delta.shape)
    # print('padded image', self.padded_img.shape)
    self.weightChanges = np.zeros(self.weights.shape)
    new_delta = np.zeros(self.input.shape)

    print('weight changes', self.weightChanges)
    increment = 0
    for batch in range(delta.shape[0]):
      for in_channels in range(self.in_channels):
        # print('bias',delta[batch,in_channels])
        # print('bias ,',np.sum(delta[batch,in_channels]))
        # print('bias shape', self.biases.shape)
        # new_delta[batch,in_channels] = self.weights

        



        for filter in range(self.weightChanges.shape[0]):
          for new_delta_row in range(delta.shape[2]-self.filterHeight+1):
            for new_delta_column in range(delta.shape[3]-self.filterWidth+1):
              new_delta[batch,in_channels,new_delta_row,new_delta_column] += np.sum(self.weights[filter]*delta[batch,in_channels+increment, new_delta_row:new_delta_row+self.filterHeight, new_delta_column:new_delta_column+self.filterWidth])
          
          
          # self.biasChanges[0,in_channels+increment] = np.sum
          

          for row in range(self.weightChanges.shape[1]):
            for column in range(self.weightChanges.shape[2]):
              #not sure if its the correct fix will need more testing/research
              self.weightChanges[filter,row,column] = np.sum(self.padded_img[batch,
                                                              in_channels,
                                                              row : self.paddedHeight - self.filterHeight + 1 + row,
                                                              column : self.paddedWidth - self.filterWidth + 1 + column] * delta[batch,
                                                              in_channels+increment,
                                                              row : self.paddedHeight - self.filterHeight + 1 + row,
                                                              column : self.paddedWidth - self.filterWidth + 1 + column] )
              # new_delta[batch,in_channels,row,column] = None
              # print(self.weightChanges[filter,row,column])
          # for row_input in range(self.input.shape[1]):
          #   for column_input in range(self.input.shape[2]):
          #     new_delta[batch,in_channels,row,column] = None
        increment += self.numberOfFilters
      increment = 0
    # print('dimensions')
    print('weight changes after', self.weightChanges)  
    print('new delta',new_delta)
    print('new delta shape', new_delta.shape)
    # print('delta,',delta)

    return new_delta

  def create_padding(self,image,padding = 'same'):
    padding_height = self.filterHeight//2
    padding_width = self.filterWidth//2
    horizontal_additional = self.filterWidth%2
    vertical_additional = self.filterHeight%2

    #additional tuning needed
    if padding == 'same':

      padded_img = np.zeros((image.shape[0],image.shape[1], image.shape[2]+2, image.shape[3]+2))

      for img in range(image.shape[0]):

        for channel in range(image.shape[1]):

          padded_img[img,channel] = np.r_[

                            np.zeros((1,image.shape[3]+2)),

                            np.c_[np.zeros((image.shape[2],1)),

                            image[img,channel],

                            np.zeros((image.shape[2],1))],

                            np.zeros((1,image.shape[3]+2))]


    return padded_img
    
  def forward(self,image):
    self.input = image
    if self.padding == True:

      self.padded_img =self.create_padding(image)
      self.paddedHeight = self.padded_img.shape[2]
      self.paddedWidth = self.padded_img.shape[3]

      #create a function that padds according tot he filter size
      #however until this is created we are going to assume that the padding is same and filter size is 3x3
    
    increment = 0
    self.zValues = np.zeros((self.padded_img.shape[0],self.out_channels,self.paddedHeight-self.filterHeight+1, self.paddedWidth-self.filterWidth+1))

    # print('filterapplied images shape',filterAppliedImages.shape)
    for images in range(self.padded_img.shape[0]):

      for in_channels in range(self.in_channels):

        for filters in range(self.numberOfFilters):

          for y in range(self.paddedHeight-self.filterHeight+1):

            for x in range(self.paddedWidth-self.filterWidth+1):

              self.zValues[images,filters+increment,x,y] = np.sum(self.padded_img[images,in_channels,y:y+self.filterHeight,x:x+self.filterWidth]*self.weights[filters])

        increment += self.numberOfFilters

      increment = 0
      
    # self.zValues = filterAppliedImages 

    if self.activation == 'relu': 
      if self.use_bias == False:
        # print('using relu')
        self.output = np.maximum(self.zValues,0)
      else:
        self.output = np.maximum(self.zValues+self.biases,0)

    if self.activation == None or 'none':
      self.output = self.zValues
