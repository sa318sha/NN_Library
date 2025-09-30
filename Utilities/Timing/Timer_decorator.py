import time
import os
from Utilities.Additional_functions.Utility_functions import *

def non_return_timer(func):
  def wrapper(*args,**kwargs):

    start_time = time.time()

    func(*args,**kwargs)

    end_time = time.time() - start_time
    
    del start_time
    # print('test',__file__)
    # print('main', os.getcwd())
    timing_file = open(os.path.join(os.getcwd(),'time.txt'), 'a')
    timing_file.write('\n' + "  time of func:" + str(end_time) + '  ' + count_args_and_kwargs_return_amount_with_arg_0(*args, **kwargs) +  '  ' )
    timing_file.close()
    
  return wrapper


def return_timer(func):
  def wrapper(*args,**kwargs):

    start_time = time.time()

    value = func(*args,**kwargs)

    end_time = time.time() - start_time
    
    del start_time

    timing_file = open(os.path.join(os.getcwd(),'time.txt'), 'a')
    timing_file.write('\n' + "  time of func:" + str(end_time) + '  ' + count_args_and_kwargs_return_amount_with_arg_0(*args, **kwargs) + '  ' )
    timing_file.close()
    return value
    
  return wrapper