from Utilities.Logging.Logging_Decorator import logger

@logger
def f(*args,**kwargs ):
  print('poop')

f(12, a =14, msg = 'logger function in utilities')