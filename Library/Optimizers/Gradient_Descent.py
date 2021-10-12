from Library.Optimizer import Optimizer
import numpy as np

class Gradient_Descent(Optimizer):
  def __init__(self, learning_rate = 0.0001):

    # self.learning_rate = learning_rate
    self.name = 'Gradient_Descent'
    super().__init__(learning_rate=learning_rate)

  def update(self,weights,biases,weightChanges,biasChanges,batch_size):
    weights -= (weightChanges*self.learning_rate)/batch_size
    biases -= (biasChanges * self.learning_rate)/batch_size
    biases = np.reshape(biases,(1,-1))
    return weights,biases