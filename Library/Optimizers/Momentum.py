import numpy as np
from Library.Optimizer import Optimizer

class Momentum(Optimizer):
  def __init__(self, learning_rate, bheta= 0.5): #bheta is the variable the controls the exponentially weighted average 
    self.bheta = bheta
    super().__init__(learning_rate=learning_rate)
  def poo():
    pass


