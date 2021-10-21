import numpy as np
np.random.seed(0)

class testClass():
  def __init__(self) -> None:
      self.test = 0
  def forward():
    pass
a = [4,12]

# a = [[24,12],[-4,2]]
# print('a', a)
test2 = np.arange(9).reshape(3,3)
print(test2)
# a = a.append(test2)
# print('a',a)
# print('test2', test2)
test3 = np.arange(12).reshape(2,6)
print('test3',test3)
# test4 = np.concatenate(test2,test3)
x = np.array([test2,test3])
print(x)
print(x[1])
# print('test4', test4,test4.shape)