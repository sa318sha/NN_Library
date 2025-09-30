# the strategy can be a number of things for our use case we will use epsilon greedy strategy
import random
import numpy as np
class Agent():
    # num actions is how many actions it can take in the enviroment
    def __init__(self,strategy,num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

        
    def select_action(self, state, policy_network):
      rate = self.strategy.get_exploration_rate(self.current_step)
      self.current_step +=1
        
      if rate > random.random():
        explore = random.randrange(self.num_actions)
#             print('explore',explore)
        return np.array([explore]) #explore

      else:      
        output = policy_network.evaluate(state) 
        return output
        #this is the epsilon greedy strategy this is the exploit we need to pass in teh states given into the policy network and then return the largest result of the network
        

#         exploit = policy_network(state)
# #        print('exploit',exploit)
#         # inputs the state and returns the largest actions
#         return exploit