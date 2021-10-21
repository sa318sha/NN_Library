class Flatten():
      
  def forward(self,input):
    self.input = input
    self.batch_shape = input.shape[0]
    self.out_channels = input.shape[1]
    self.height_shape = input.shape[2]
    self.width_shape = input.shape[3]


    self.output = input.reshape(input.shape[0], -1)

  def backPropogation(self,delta):
    return delta.reshape(delta.shape[0],self.out_channels,self.height_shape,self.width_shape)
    
    