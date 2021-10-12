

import numpy as np

class Layer_Dense():
  def __init__(self,n_inputs,n_neurons,activation = 'relu'):
    self.activation = activation
    self.neuronsShape = n_neurons
    self.inputsShape = n_inputs
    
    self.weights = np.random.randn(n_neurons,n_inputs) # randn(i,j) creates random shape of rows i and columns j with normal distrubution where i is the number of neurons in current layer and j is number of neurons in previous layer or input
    # we are diubg inputs first and then neurons second because, although it should be the other way around, this way we dont need to transpose when doing the dot product so saves computational speed
    self.biases = np.zeros((1,n_neurons))
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

  def backPropogation(self,target,costFunc,output = True):
    
    pass


  def forward(self,inputs):

    self.inputs = inputs
    self.activation = self.activation
    self.zValues = np.dot(inputs,self.weights.T) + self.biases

    if self.activation == 'relu': 

      self.output = np.maximum(0,self.zValues)

    elif self.activation == 'softmax':
      exp_values = np.exp(self.zValues -np.max(self.zValues,axis=1,keepdims=True))
      norm_values = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
      self.output = norm_values

    elif self.activation == 'none':
      pass
    else:
      print('poo')

