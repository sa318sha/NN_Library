import numpy as np

class EpsilonGreedyStrategy():
    def __init__(self, minimum, maximum, decay):
        self.min = minimum
        self.max = maximum
        self.decay = decay
        
    def get_exploration_rate(self,current_step):
        return self.min +(self.max - self.min) * np.exp(-1 * current_step*self.decay)