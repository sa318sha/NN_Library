import numpy as np
from Library.Optimizer import Optimizer






class Adam(Optimizer):
  def __init__(self, learning_rate = 0.0001,bheta1 = 0.9,bheta2 = 0.999, epsilon = 10e-8):
    self.learning_rate = learning_rate
    self.bheta1 = bheta1
    self.bheta2 = bheta2
    self.epsilon = epsilon
    self.name = 'Adam'
    print(epsilon)
    super().__init__(learning_rate=learning_rate)


  def update(self,V_dW, V_db,S_dW, S_db, weightChanges, biasChanges,batch_size,weights, biases,t):
    t+=1
    # print('weights shape',weights.shape)
    # print('V_dw b4', V_dW)
    # print(self.bheta1,'v_dw without addition',(1-self.bheta1)*weightChanges)
    # print('S_dw before change',S_dW)
    
    V_dW = self.bheta1*V_dW + (1-self.bheta1)*weightChanges
    V_db = self.bheta1*V_db + (1-self.bheta1)*biasChanges
    S_dW = self.bheta2*S_dW + (1-self.bheta2)*np.square(weightChanges)
    S_db = self.bheta2*S_db + (1-self.bheta2)*np.square(biasChanges)
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
    weights -= self.learning_rate*(V_dW/(np.sqrt(S_dW)+self.epsilon))
    biases -= self.learning_rate*(V_db/(np.sqrt(S_db)+self.epsilon))
    biases = np.reshape(biases,(1,-1))
    # print('right before function finish',weights)
    return V_dW,V_db,S_dW,S_db, t
    

    # self.V_dW,self.V_dB,self.S_dW,self.S_dB,self.weights, self.biases

# Adam.update = classmethod(Adam.update)
# classmethod()
