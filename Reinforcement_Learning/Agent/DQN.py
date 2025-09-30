from Library.Models.Sequential import Sequential
from Library.Optimizers.Optimizer import Optimizer
from Library.metrics.Metrics import metrics
from Library.Optimizers.Adam import Adam
from Library.Layers.Layer_Dense import Layer_Dense
from Library.Layers.Convolutional_Layer import Convolutional_Layer
from Library.Layers.MaxPool2D import MaxPool2D
from Library.Layers.Flatten import Flatten
from Image_Manipulation.image_data_generator import ImageDataGenerator
import numpy as np
import pandas as pd

class DQN():
  def __init__(self,height,width,model,memory) -> None:
    
    self.