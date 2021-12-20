import gym
import numpy as np

class CartPoleEnvManager():
    def __init__(self,device):
        # self.device = device
        self.env = gym.make('CartPole-v1').unwrapped
        #unwrapped gives us access to backend dynamics of the environemnt we wouldnt be able to use otherwize
        self.env.reset()
        #current screen will track the current screen of the environemnt at any given time 
        self.current_screen = None
        #done if any actiont aken finishes the episode or if the episode closes
        self.done = False
#         self.env.step??
        
    #reset close and render are gym function but since we are wrapping the gym environemnt we need to make our own
    def reset(self):
        self.env.reset()
        #we reset current_screen here because usually we want to reset it at the end of the episode and usually fym doesnt do this for us
        self.current_screen = None
        
    def close(self):
        self.env.close()
        
    def render(self, mode = 'human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self,action):
#         print('action',action,action.item())
        state, reward, self.done, info = self.env.step(action.item())
        return np.array([reward])

            
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            #since we are saying that a single state is the difference between the current and previous state we initialize the starting as a black screen
            #similarily we say the final screen is black when its finished to prepare for the next episode
            self.current_screen = self.get_processed_screen()
            black_screen = np.zeros(self.current_screen)
            return black_screen
        
        else:
            #s1 is current screen and s2 is the next screen to get we call get processed_screen
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2-s1
        
        
    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]
    
    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]
    
    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2,0,1))
        #expects in CHW
        #renders as rgb arrawy and then transposes in the order of channel,height,width
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self,screen):
        screen_height = screen.shape[1]
        #tjere is unnecassry stuff in the cartpole screen
        top = int(screen_height *0.4)
        bottom = int(screen_height * 0.8)
        screen = screen [:, top:bottom, :]
        return screen
    
    def transform_screen_data(self,screen):
        #convert to float rescale and convert to tensor
        screen = np.ascontiguousarray(screen,dtype = np.float32)/255
        #torchvision - > defined a certain transformation and any input will go through those image transformations  
        #PIL is a python library for image manipulation
        # resize = T.Compose([
        #     T.ToPILImage(),
        #     T.Resize((40,90)),
        #     T.ToTensor()
        # ])
        return resize(screen).unsqueeze(0).to(self.device)