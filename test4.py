import numpy as np

delta=np.array([2.5,-1.3])

weights = np.arange(20).reshape(2,10)
print('delta',delta)
print('weights',weights)

print('delta transposed',delta.T)
print('weights transposed', weights.T)

new_delta = np.sum(delta*weights.T,axis=1)

print('multiply', new_delta)