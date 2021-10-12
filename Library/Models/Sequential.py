import numpy as np
from Library.Model import Model

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
    #might want to change backpropogation for each layer and how instead of 
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

