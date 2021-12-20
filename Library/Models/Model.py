
from math import remainder
from os import pardir
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
  
  # def params(self):
  #   parameter = np.array([], dtype=object)
  #   for layer in self.Layer:
  #     print('parameter', parameter)
  #     print('layer wights', layer.weights)

  #     parameter = np.array([parameter,layer.weights],dtype=object)
  #   print('parameter',parameter)
  #   return parameter


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
        # print('mini batch',mini_batch)
        print('mini batch',mini_batch)
        self.forward(shuffled_training_input[range(startingIndex,startingIndex+batch_size)])
        self.backPropogation(shuffled_training_target[range(startingIndex,startingIndex+batch_size)])
        startingIndex += batch_size
      # print('in model checking the input and target respectivly shape', shuffled_training_input.shape,shuffled_training_target.shape)
      self.forward(shuffled_training_input[range(startingIndex,int(len(training_target)))])
      self.backPropogation(shuffled_training_target[range(startingIndex,int(len(training_target)))])

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
      # print(self.metric_values)
      for metric in self.metrics:
        self.metric_values.append(metric(self.output,shuffled_training_target))
      self.evaluation(testing_input,testing_target)
      self.display_results(epoch)
      


      
  def evaluation(self,testing_input,testing_target):
    self.forward(testing_input)
    self.test_loss_value = self.loss_function(self.output,testing_target)
    for metric in self.metrics:
        self.test_metric_values.append(metric(self.output,testing_target))

    
  def compile(self,loss_function,metrics):



    self.loss_function = loss_function
    self.metrics = metrics #metrics is stored in an array
    

  def display_results(self,epoch = -1):
    print('\n----------------------------------------------')
    print('epoch:', epoch+1 if epoch != -1 else 'Final Results')
    print('training loss', np.mean(self.loss_value))
    print('validation loss', np.mean(self.test_loss_value))
    # print(self.metric_values)
    print('accuracy', np.mean(np.array(self.metric_values)))
    print('validation accuracy', np.mean(np.array(self.test_metric_values)))
    print('----------------------------------------------\n')
    # self.metric_values = []

