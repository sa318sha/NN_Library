
from math import remainder
import numpy as np
import random

from numpy.core.numeric import NaN

class Model:
  def __init__(self,Layers):
    self.Layer = Layers
    self.metric_values = []
    self.test_metric_values = []
  def check(self):
    pass
  
  def predict(self):
    pass

  def fit(self,epochs,batch_size,input,target, validation_split = 0.0):
    self.validation_split = validation_split
    use_batch_size = True
    
    shuffled_input = np.empty(input.shape, dtype=input.dtype)
    shuffled_target = np.empty(target.shape, dtype=target.dtype)
    permutation = np.random.permutation(len(input))
    for old_index, new_index in enumerate(permutation):
      shuffled_input[new_index] = input[old_index]
      shuffled_target[new_index] = target[old_index]

    training_input = np.array(shuffled_input[int(len(shuffled_input) * (validation_split)): ])
    training_target = np.array(shuffled_target[int(len(shuffled_target) * (validation_split)): ])
    testing_input = np.array(shuffled_input[:int(len(shuffled_input)*validation_split)])
    testing_target = np.array(shuffled_target[:int(len(shuffled_target)*validation_split)])
    
    
    
    
    numberOfIterations = int(len(training_target)/batch_size)
    remainder = int(len(training_target)%batch_size)
    print('number of iterations',numberOfIterations)
    print('batch size', batch_size)
    print('data length',len(training_target))
    print('remainder', remainder)




    for epoch in range(epochs): #apperently mini batch you do gradient descent for the whole data set but where the batch size is teh data set for the descent 
      
      shuffled_training_input = np.empty(training_input.shape, dtype=training_input.dtype)
      shuffled_training_target = np.empty(training_target.shape, dtype=training_target.dtype)
      permutation = np.random.permutation(len(training_input))
      for old_index, new_index in enumerate(permutation):
        shuffled_training_input[new_index] = training_input[old_index]
        shuffled_training_target[new_index] = training_target[old_index]
      
      
      
      
      startingIndex = 0




      for mini_batch in range(numberOfIterations):
        print('mini batch',mini_batch)
        self.forward(shuffled_training_input[range(startingIndex,startingIndex+batch_size)])
        self.backPropogation(shuffled_training_target[range(startingIndex,startingIndex+batch_size)])
        startingIndex += batch_size
        for layer in self.Layer:
          layer.layer_update(batch_size,self.optimizer)

      
      self.forward(shuffled_training_input[range(startingIndex,int(len(training_target)))])
      self.backPropogation(shuffled_training_target[range(startingIndex,int(len(training_target)))])
      for layer in self.Layer:
        layer.layer_update((int(len(training_target)-startingIndex)),self.optimizer)

      # training_input[range(startingIndex,int(len(training_target)))]
      # for i in range(batch_size):
      #   pass
      # if(use_batch_size == True): 
      #   batch_target = np.array(random.sample(training_target.tolist(),batch_size))

      # self.forward(training_input)
      # # self.getFinalOutput()
      # self.backPropogation(training_target if use_batch_size == False else batch_target)
      
      # for layer in self.Layers:
      #   layer.layer_update(self.learning_rate,batch_size)
      # # print('output', self.output)
      # # want to edit this so onece layer is updated the validation dataset is evaluated and it metrics and loss functions are displayed as well as the training
      self.forward(shuffled_training_input)
      self.loss_value = self.loss_function(self.output,shuffled_training_target)
      for metric in self.metrics:
        self.metric_values.append(metric(self.output,shuffled_training_target))
      self.evaluation(testing_input,testing_target)
      self.display_results(epoch)
      


      
  def evaluation(self,testing_input,testing_target):
    self.forward(testing_input)
    self.test_loss_value = self.loss_function(self.output,testing_target)
    for metric in self.metrics:
        self.test_metric_values.append(metric(self.output,testing_target))

    
  def compile(self,loss_function,metrics, optimizer):
    self.optimizer = optimizer

    self.loss_function = loss_function
    self.metrics = metrics #metrics is stored in an array
    

  def display_results(self,epoch = -1):
    print('\n----------------------------------------------')
    print('epoch:', epoch+1 if epoch != -1 else 'Final Results')
    print('training loss', np.mean(self.loss_value))
    print('validation loss', np.mean(self.test_loss_value))
    print('accuracy', np.mean(np.array(self.metric_values)))
    print('validation accuracy', np.mean(np.array(self.test_metric_values)))
    print('----------------------------------------------\n')
    

class Sequential(Model):
  
  def __init__(self,Layers):
    # Layers parameter would ideally be an array ofprev layerswould each Layer
    self.Layers = Layers
    self.forwardCalled = False
    
    for i in Layers:
      print('initialization',i.weights)
    super().__init__(Layers)
  def testingFunc(self):
    for layer in self.Layers:
      # print(layer.weights)
      pass
    print(self.learning_rate)
  
  def forward(self,input):
    temp = input.copy()
    self.forwardCalled = True
    for layer in self.Layers:

      layer.forward(input)
      input = layer.output
      self.output = layer.output
  
    self.input = temp


  def getFinalOutput(self):
    # if(self.forwardCalled == False):

    self.output = self.Layers[len(self.Layers)-1].output # may need to change weights to output once forward prop is finished

  def backPropogation(self,target):
    #propogation attempt 2
    if len(target.shape) == 2:
      target = np.argmax(target,axis=1) # hot encodes the target

    outputLayer = True
    count = 0
    for p in target: 
      # print('count',count)
      # print('target', target)
      y = p
      outputLayer = True
      oldWeights = 0
      # print(target.shape)
      oldDelta=0
      

      for layer in reversed(self.Layers):
        
        delta = np.zeros(layer.weights.shape[0])
        ChangeInWeight = np.zeros(layer.weights.shape)
        # print('change in weight', ChangeInWeight)
        if(outputLayer == True):
          outputLayer = False
          if(layer.activation == 'softmax'):


            for row in range(layer.weights.shape[0]): #itterates throught the rows
              if row == y:
                delta[row] = (layer.output[count,row]-1)
              elif row != y: 

                delta[row] = (layer.output[count,row])

              for column in range(layer.weights.shape[1]):
                # print(layer.inputs)
                ChangeInWeight[row,column] = delta[row]*layer.inputs[count,column]


            oldDelta = delta.copy()
            oldWeights = layer.weights

            layer.weightChanges += ChangeInWeight
            layer.biasChanges += delta

            
            
          elif(layer.activation == 'relu'):
            pass
          
        else:
          if(layer.activation == 'softmax'):
            pass
          elif(layer.activation == 'relu'):


            for row in range(layer.weights.shape[0]):
              
              sum = np.sum(oldDelta * oldWeights[:,row])

              if(layer.zValues[0,row] > 0):
                delta[row] = sum

              elif(layer.zValues[0,row] < 0):
                delta[row] = 0

              else:
                delta[row] = sum*1/2



              for column in range(layer.weights.shape[1]): #there are 4 columns

                ChangeInWeight[row,column] = delta[row] * layer.inputs[count,column]

            layer.weightChanges += ChangeInWeight
            layer.biasChanges += delta
            oldWeights = layer.weights
            oldDelta = delta.copy()


          elif(layer.activation == 'sigmoid'):
            pass
      count+=1

