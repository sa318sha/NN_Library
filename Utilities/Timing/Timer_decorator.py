import time
from Utilities.Additional_functions.Utility_functions import count_args_and_kwargs_return_string

def non_return_timer(func):
  def wrapper(*args,**kwargs):

    start_time = time.time()

    func(*args,**kwargs)

    end_time = time.time - start_time
    del start_time

    timing_file = open(r'C:\Users\kobru\OneDrive\Desktop\Personnel_Projects\NN_library\time.txt', 'a')
    timing_file.write('\n' + "  time of func:" + end_time + '  ' + count_args_and_kwargs_return_string(*args, **kwargs))
    timing_file.close()
    
  return wrapper