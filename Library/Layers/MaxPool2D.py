import numpy as np
from numpy.core.fromnumeric import shape
from Utilities.Logging.Logging_Decorator import *
from Utilities.Timing.Timer_decorator import non_return_timer, return_timer
# from Utilities.Timing.Timer_decorator import non_return_timer



np.random.seed(0)
class MaxPool2D():
  def __init__(self,pool_size,stride) -> None:
      self.stride = stride
      self.pool_height, self.pool_width = pool_size

  

  @non_return_timer
  @non_return_logger
  def forward(self,image, forward = True):
    print('pool layer forward')
    self.input = image
    self.zValues = np.zeros((image.shape[0],image.shape[1],int(np.ceil(image.shape[2]/self.stride)),int(np.ceil(image.shape[3]/self.stride))))
    # print('zvalues',self.zValues)
    # print('zvalues shape', self.zValues.shape)
    # print('image', image)
    # print('image shape', image.shape)
    rowCounter = 0
    columnCounter =0
    for batch in range(image.shape[0]):
      for filter in range(image.shape[1]):
        rowCounter =0
        for row in range(0,image.shape[2],self.stride):
          columnCounter = 0
          for column in range(0,image.shape[3],self.stride):
            if(row + self.pool_height < image.shape[2] and column +self.pool_width < image.shape[3]):
              self.zValues[batch,filter,rowCounter,columnCounter] = np.max(self.input[batch,filter,row: row+ self.pool_height, column : column + self.pool_width])
              # print(result[1])

            if(row + self.pool_height > image.shape[2] and column +self.pool_width < image.shape[3]):
              self.zValues[batch,filter,rowCounter,columnCounter] = np.max(self.input[batch,filter,row:, column : column + self.pool_width])

            if(row + self.pool_height < image.shape[2] and column +self.pool_width > image.shape[3]):
              self.zValues[batch,filter,rowCounter,columnCounter] = np.max(self.input[batch,filter,row: row+ self.pool_height, column :])

            if(row + self.pool_height > image.shape[2] and column +self.pool_width > image.shape[3]):
              self.zValues[batch,filter,rowCounter,columnCounter] = np.max(self.input[batch,filter,row:, column :])

            columnCounter +=1
          rowCounter+=1



    # print('zValues at the end', self.zValues)
    self.output = self.zValues
  # @non_return_timer
  @return_timer
  def backPropogation(self,delta):
    print('back propping maxPool 2d')
    new_delta = np.zeros(self.input.shape)
    # print('new delta initialized',new_delta)
    for batch in range(delta.shape[0]):
      for filter in range(delta.shape[1]):

        for row in range(delta.shape[2]):
          for column in range(delta.shape[3]):
            if(row*self.stride + self.pool_height < self.input.shape[2] and column*self.stride +self.pool_width < self.input.shape[3]):
              result = np.where(self.zValues[batch,filter,row,column] == self.input[batch,filter,row*self.stride:row*self.stride+self.pool_height,column*self.stride: column*self.stride + self.pool_width])
              # print('is this case run when it shouldnt be?')

            elif(row*self.stride + self.pool_height > self.input.shape[2] and column*self.stride +self.pool_width < self.input.shape[3]):
              result = np.where(self.zValues[batch,filter,row,column] == self.input[batch,filter,row*self.stride:,column*self.stride: column*self.stride + self.pool_width])
              # print('case 1')
              # print(result[0]+row*self.stride,result[1]+column*self.stride)
              

            elif(row*self.stride + self.pool_height < self.input.shape[2] and column*self.stride +self.pool_width > self.input.shape[3]):
              result = np.where(self.zValues[batch,filter,row,column] == self.input[batch,filter,row*self.stride:row*self.stride+self.pool_height,column*self.stride:])
              # print('case 2')
              # print(result[0]+row*self.stride,result[1]+column*self.stride)

            elif(row*self.stride + self.pool_height > self.input.shape[2] and column*self.stride +self.pool_width > self.input.shape[3]):
              result = np.where(self.zValues[batch,filter,row,column] == self.input[batch,filter,row*self.stride:,column*self.stride:])
              # print(result[0]+row*self.stride,result[1]+column*self.stride)
              # print('ding gind gindfs')
            # # print(self.zValues[batch,filter,row,column])
            # result = np.where(self.zValues[batch,filter,row,column] == self.input[batch,filter])
            # # print('location',new_delta[batch,filter,result[0],result[1]])
            # print(result[0]+row*self.stride,result[1]+column*self.stride)
            # print('shape',new_delta.shape)
            new_delta[batch,filter,result[0]+row*self.stride,result[1]+column*self.stride] = delta[batch,filter,row,column]
            # new_delta[batch,filter,result[0],result[1]] = delta[batch,filter,row,column]

            
            

    # for batch
    # print('new delta',new_delta)
    return new_delta



