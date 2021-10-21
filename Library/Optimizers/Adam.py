import numpy as np
from Library.Optimizers.Optimizer import Optimizer






class Adam(Optimizer):
  def __init__(self, learning_rate = 0.0001,bheta1 = 0.9,bheta2 = 0.999, epsilon = 10e-8):
    self.learning_rate = learning_rate
    self.bheta1 = bheta1
    self.bheta2 = bheta2
    self.epsilon = epsilon
    self.V_dW = 0
    self.V_db = 0
    self.S_dW = 0
    self.S_db = 0
    self.t = 0
    self.name = 'Adam'
    print(epsilon)

    self.check = np.random.randint(4)

    super().__init__(learning_rate=learning_rate)


  def update(self, weightChanges, biasChanges):
    self.t+=1
    # print('check', self.check)
    # print('weights shape',weights.shape)
    # print('V_dw b4', V_dW)
    # print(self.bheta1,'v_dw without addition',(1-self.bheta1)*weightChanges)
    # print('S_dw before change',S_dW)
    
    self.V_dW = self.bheta1*self.V_dW + (1-self.bheta1)*weightChanges
    self.V_db = self.bheta1*self.V_db + (1-self.bheta1)*biasChanges
    self.S_dW = self.bheta2*self.S_dW + (1-self.bheta2)*np.square(weightChanges)
    self.S_db = self.bheta2*self.S_db + (1-self.bheta2)*np.square(biasChanges)
    # print('S_dw after change',S_dW)
    # print('V_dw after', V_dW)
    # correction
    #im going to try without correction because the correction makes the corrected value skyrocket by having lets say 0.5 to the power of 20
    # V_dW_corrected = V_dW /(1-(self.bheta1**t))
    # V_db_corrected = V_db /(1-(self.bheta1**t))
    # S_dW_corrected = S_dW /(1-(self.bheta2**t))
    # S_db_corrected = S_db /(1-(self.bheta2**t))
    # print('corrected S_dw after',S_dW_corrected)
    # print('v_DW', V_dW)
    # print('power',self.bheta2**t)
    # print('before wieght change',weights)
    # print('change factor', self.learning_rate*(V_dW_corrected/(np.sqrt(S_dW_corrected)+self.epsilon)))
    Changedweights = self.learning_rate*(self.V_dW/(np.sqrt(self.S_dW)+self.epsilon))
    Changedbiases = self.learning_rate*(self.V_db/(np.sqrt(self.S_db)+self.epsilon))
    # print('bias changes shape before', Changedbiases.shape)
    # Changedbiases = np.reshape(Changedbiases,(1,-1))
    return Changedweights,Changedbiases
    # print('right before function finish',weights)
    # return V_dW,V_db,S_dW,S_db, t
    

    # self.V_dW,self.V_dB,self.S_dW,self.S_dB,self.weights, self.biases

# Adam.update = classmethod(Adam.update)
# classmethod()
