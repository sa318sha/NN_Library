
def extract_args_and_kwargs_return_string(*args,**kwargs):
  temp = ' '
  for i in args:
    
    temp = temp + str(i) + ' '
    # print(temp)

  
  for k in kwargs:
    temp = temp + str(k) + ' = '+ str(kwargs[k]) + ' '  
    # print(temp)
  return temp

def count_args_and_kwargs_return_string(*args,**kwargs):
  return 'Amount of args:' + str(len(args)) + ' and kwargs: ' + str(len(kwargs))