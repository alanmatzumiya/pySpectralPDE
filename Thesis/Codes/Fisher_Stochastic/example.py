# importing the required modules 
import timeit 
from main import run_burgers
  
def time_function(): 
    return run_burgers()

print(timeit.timeit(time_function)) 

