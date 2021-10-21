from os import error
import numpy as np
from Utilities.Logging.Logging_Decorator import return_logger
# fr import Adam, Gradient_Descent,Mini_batch_gradient_descent,Momentum

class Layer_Dense():
  def __init__(self,n_inputs,n_neurons,activation = 'relu',optimizer = None):
    self.optimizer = optimizer
    self.activation = activation
    self.neuronsShape = n_neurons
    self.inputsShape = n_inputs
    self.output_layer = None
    self.weights = np.random.randn(n_neurons,n_inputs) # randn(i,j) creates random shape of rows i and columns j with normal distrubution where i is the number of neurons in current layer and j is number of neurons in previous layer or input
    # we are diubg inputs first and then neurons second because, although it should be the other way around, this way we dont need to transpose when doing the dot product so saves computational speed
    self.biases = np.zeros((1,n_neurons))
    self.weightChanges = self.weights.copy()
    self.biasChanges = self.biases.copy()


  def set_weights(self,newWeights):

    self.weights = newWeights

  def set_bias(self,newBias):

    self.biases = newBias

  def set_values(self,newBias,newWeights):

    self.weights = newWeights
    self.biases = newBias

  # def layer_update(self,batch_size,optimizer):

  #   if(optimizer.name == 'Adam'):


  #     # print('weights b4', self.weights)
  #     # print('before',self.weights)

  #     self.V_dW,self.V_dB, self.S_dW, self.S_dB, self.t = optimizer.update(self.V_dW, self.V_dB,self.S_dW, self.S_dB, self.weightChanges,self.biasChanges,batch_size,self.weights,self.biases,self.t)
  #     # print("T IS", self.t)
  #     # print('after',self.weights)
  #     # print('V_dw after optimizer in this layer', self.V_dW)
  #     # print('after function finish S_dw',self.S_dW)
  #     # print('weights after', self.weights)
  #     # print('weights shape', self.weights.shape)


  #   elif(optimizer.name == 'Gradient_Descent'):
  #     self.weights, self.biases = optimizer.update(self.weights,self.biases,self.weightChanges,self.biasChanges,batch_size)

    
  #   else:
  #     print('no optimizer used')

  
  def backPropogation(self,delta,target = None,output_layer = False):

    
    new_delta = np.zeros((self.inputs.shape))
   
    if (output_layer == True):
      count = 0
      # print('target')
      for p in target:

        for a in range(delta.shape[1]):

          
          if self.activation == 'softmax':
            if(p==a):
              delta[count,a] = self.output[count,a]-1
            else:
              delta[count,a] = self.output[count,a]

        count+=1
    

    for batch in range(delta.shape[0]):

      for row in range(self.weights.shape[0]):

        for column in range(self.weights.shape[1]):

          if self.activation == 'relu' and output_layer == False:
            if self.zValues[batch,row] < 0:
              delta[batch,row] =0
          # elif self.activation == 'softmax':
          #   pass
          # elif self.activation == 'sigmoid':
          #   pass
          # else:
          #   pass

          self.weightChanges[row,column] += delta[batch,row]*self.inputs[batch,column]
          
      new_delta[batch] = np.sum(delta[batch,:] * self.weights.T, axis=1)
      self.biasChanges += delta[batch]
      # print(self.weights)
      
      # print(self.weights)
    weightChanges, biasChanges = self.optimizer.update(self.weightChanges,self.biasChanges)
    self.weights -= weightChanges
    self.biasChanges -= biasChanges
    return new_delta

       
  def forward(self,inputs):

    self.inputs = inputs
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

