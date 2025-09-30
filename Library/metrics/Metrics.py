import numpy as np

class metrics:

  def forward(self,y_output,y_target):

    if(len(y_target.shape) ==2):
      val = (y_target[range(len(y_target)), np.argmax(y_output,axis=1)])
    elif len(y_target.shape) == 1 and type(y_output) == 'list':
      val = (y_target[range(len(y_target)), y_output])

    return val
  def categorical_crossEntropy(y_output,y_target):
    # print('in categorical crossEntropy y_output and y_target are,', y_output,y_target)
    y_output_clipped = np.clip(y_output, 1e-7, 1 -1e-7)
    if len(y_target.shape) == 1:
      # print(y_target)
      loss_array = y_output_clipped[range(len(y_output)), y_target]
      # print('loss array', loss_array, y_target)
    elif len(y_target.shape) == 2:
      # print(np.argmax(y_target, axis=1))
      # print(y_output)
      loss_array = y_output_clipped[range(len(y_output)), np.argmax(y_target, axis=1)]
    negative_log = -np.log(loss_array)

    return negative_log
  def accuracy(y_output,y_target):
    acc = 0
    print('shape ', y_target.shape)
    if(len(y_target.shape) ==2):
      acc = (y_target[range(len(y_target)), np.argmax(y_output,axis=1)])
    # elif len(y_true.shape) > 2 :
    #   print('yes pls')
      
    elif(len(y_target.shape)==1):
      acc = np.rint(y_output[range(len(y_output)), y_target])
      # print(acc)      # print('max', np.argmax(y_output,axis = 1).shape)
      # print(y_output)
      # acc = np.argmax(y_output,axis=1)[range(len(y_output))]

    return acc