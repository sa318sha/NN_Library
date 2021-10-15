

from os import error
import numpy as np

class Layer_Dense():
  def __init__(self,n_inputs,n_neurons,activation = 'relu'):
    self.activation = activation
    self.neuronsShape = n_neurons
    self.inputsShape = n_inputs
    self.output_layer = None
    self.weights = np.random.randn(n_neurons,n_inputs) # randn(i,j) creates random shape of rows i and columns j with normal distrubution where i is the number of neurons in current layer and j is number of neurons in previous layer or input
    # we are diubg inputs first and then neurons second because, although it should be the other way around, this way we dont need to transpose when doing the dot product so saves computational speed
    self.biases = np.zeros((1,n_neurons))
    self.weightChanges = self.weights.copy()
    self.biasChanges = self.biases.copy()
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

  def backPropogation(self,delta,target = None,output_layer = False):

    # pseudo code
    # if self.output_layer == True:
    #   if self.activation
    # if(self.activation == 'softmax' )




    
    # print('output inside ', self.output)
    
    # print('delta', delta)
    # print('delta shape', delta.shape)
    # print('first delta', delta[0])
    # print('input',self.inputs)
    # print('inputs shape', self.inputs.shape)

    # print('change in weight',self.weightChanges)
    # print('weights shape',self.weights.shape)
    new_delta = np.zeros((self.inputs.shape))
    # print('detla',delta[0,:])
    # print('input at 6',self.inputs[0,5])
    # print('multiplication', delta[0,:]* self.inputs[0,0,:])
    # # print(temp)
    # print('delta',delta)
    # print('weights',self.weights)
    # print('single delta', delta[0,:])
    # print('delta shape', delta.shape[0])
    # print('single weight', self.weights[:,0])
    # new_delta = np.zeros((delta.shape[0],self.weights.shape[1]))
    # print(new_delta)
    # for column in range(self.weights.shape[1]):
    #   print(column)
    #   multiplied = np.sum(delta[0,:] * self.weights[:,column])

    # print('zvalues',self.zValues)
    # # print('multiplied',multiplied)
    # # sum = np.sum(multiplied)
    # # print(sum)
    # print('output', self.output)
    # print('target',target)
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
    else: #not output layer
      if self.activation == 'relu':
        for row in range(delta.shape[0]):

          for column in range(delta.shape[1]):

            if self.zValues[row,column] < 0:
              delta[row,column] = 0
        # print('relu activated')
        # print('zvalues', self.zValues)

    # print('delta',delta)

    # print('zvalues',self.zValues)
    # print('zvalues shape',self.zValues.shape)
    # print('coulumnmn range', self.weights.shape[1])
    # print('delta', delta)
    # print('delta shape', delta.shape)
    # print('column iterable',self.weights.shape[1])
    # print('row iterable', self.weights.shape[0])
    # print('zvalue shapes',self.zValues.shape)
    # print('zvalue', self.zValues)
    # print('weights',self.weights)
    # print('weights transposed', self.weights.T)
    for batch in range(delta.shape[0]):
      self.biasChanges += delta[batch]
      # print('the multiplication', delta[batch,:] * self.weights.T)
      # print('batch', batch)
      new_delta[batch] = np.sum(delta[batch,:] * self.weights.T, axis=1)
      for row in range(self.weights.shape[0]):

        # print('whole z values',self.zValues[batch])
        # print('total length', self.weights.shape[1], 'row length', self.weights.shape[0])
        
          # print(column,self.zValues[batch,column])
          # if self.zValues[batch,column] <= 0:
          #   delta[batch,column] = 0
          # else:
          #   pass
        
        # new_delta[batch,column] = np.sum(delta[batch,:] * self.weights[:,column])

        for column in range(self.weights.shape[1]):
          # print()


          self.weightChanges[row,column] += delta[batch,row]*self.inputs[batch,column]
          pass
        # print(row,delta[:,row])
    # print('weightchnages', self.weightChanges)
    # print('new Delta', new_delta,new_delta.shape)

    # print('final new delta', new_delta)
    # print('new delta',new_delta,new_delta.shape)
    return new_delta

          
          
    #       pass
    #       # self.weightChanges[i,j] += delta[z,i]*self.inputs[z,j]
    #       # print('i,j',i,j)
    # print(new_delta)
    # print('activation',self.activation)
    # # print('change in weight',self.weightChanges)


    # 

    #delta is a x by y array where x denotes the delta values of a certain input and y denotes the amount of 
    # print('delta',delta)

    # Ty?peError
    # pass
  

# 
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

