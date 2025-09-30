class Flatten():
      
  def forward(self,input):
    self.input = input
    self.batch_shape = input.shape[0]
    self.out_channels = input.shape[1]
    self.height_shape = input.shape[2]
    self.width_shape = input.shape[3]

    
    
    self.output = input.reshape(input.shape[0], -1)
    print('flatten foward')
    # print(self.output)
    # print(self.output.shape)
  def backPropogation(self,delta):
    print('flatten backward')
    return delta.reshape(delta.shape[0],self.out_channels,self.height_shape,self.width_shape)
    
    