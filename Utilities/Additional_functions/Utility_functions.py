
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

def count_args_and_kwargs_return_amount_with_arg_0(*args,**kwargs):
  temp = ''
  temp += str(args[0]) + '  ' 
  # print('args',len(args))
  temp += 'number of args:  ' + str(len(args)) + '  '
  temp += 'number of kwargs:  ' + str(len(kwargs))
    # print(temp)
  return temp

