import numpy as np
from numpy.core.fromnumeric import reshape
np.random.seed(0)


weigth_width =3
weight_height = 3
delta_height = 5
delta_width = 5
weight = np.random.randn(weight_height,weigth_width)
delta = np.random.randn(delta_height,delta_width)
new_delta = np.zeros((5,5))
print('weight',weight)
print('delta',delta)
print()

padded_delta = np.zeros((delta_height+2,delta_width+2))
print(padded_delta)
padded_delta  = np.r_[np.zeros((1,delta.shape[1]+2)),np.c_[np.zeros((delta.shape[0],1)), delta, np.zeros((delta.shape[0],1))], np.zeros((1,delta.shape[1]+2))]
print(padded_delta)
print(new_delta)

padded_delta_height = 7
padded_delta_width = 7

for row in range(padded_delta_height-weight_height+1):

  for column in range(padded_delta_width-weigth_width+1):

    # print(delta[row:row+weight_height, column:column+weigth_width])
    new_delta[row,column] = np.sum(weight*padded_delta[row:row+weight_height, column:column+weigth_width])
    # print(row,column)
# result = delta*weight
print(new_delta)

print('new delta',new_delta)