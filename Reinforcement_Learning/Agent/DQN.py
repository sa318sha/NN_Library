class DQN():
  def __init__(self,model,memory) -> None:
    super.__init__()
    self.model = model
    self.memory = memory
