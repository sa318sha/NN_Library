from datetime import  datetime
from Utilities.Additional_functions.Utility_functions import *
import os

def return_logger(func):
  def wrapper(*args, **kwargs):
    
    current_time =  datetime.now().strftime("Date (D/M/Y): %D Time: %H:%M:%S")

    logging_file = open(os.path.join(os.getcwd(),'time.txt'), 'a')
    logging_file.write('\n' + current_time + "  Log:"+ extract_args_and_kwargs_return_string(*args,**kwargs))          
    
    logging_file.close()

    return func(*args, **kwargs)

  return wrapper

def non_return_logger(func):
  def wrapper(*args, **kwargs):
    
    current_time =  datetime.now().strftime("Date (D/M/Y): %D Time: %H:%M:%S")

    logging_file = open(os.path.join(os.getcwd(),'time.txt'), 'a')
    logging_file.write('\n' + current_time + "  Log:"+ extract_args_and_kwargs_return_string(*args,**kwargs))          
    
    logging_file.close()

    func(*args, **kwargs)

  return wrapper



# same as weriting
# x = f1(f)
# x()