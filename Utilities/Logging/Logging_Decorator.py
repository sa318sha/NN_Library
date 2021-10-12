from datetime import  datetime

def extract_args_and_kwargs_return_string(*args,**kwargs):
  temp = ' '
  for i in args:
    
    temp = temp + str(i) + ' '
    # print(temp)

  
  for k in kwargs:
    temp = temp + str(k) + ' = '+ str(kwargs[k]) + ' '  
    # print(temp)
  return temp

def logger(func):
  def wrapper(*args, **kwargs):
    
    current_time =  datetime.now().strftime("Date (D/M/Y): %D Time: %H:%M:%S")

    logging_file = open(r'C:\Users\kobru\OneDrive\Desktop\Web Development\NN_library\log.txt', 'a')
    logging_file.write('\n' + current_time + "  Log:"+ extract_args_and_kwargs_return_string(*args,**kwargs))          
    
    logging_file.close()

    func(*args, **kwargs)

  return wrapper



# same as weriting
# x = f1(f)
# x()