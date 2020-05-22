import timeit
from main import run_burgers


def time_function():

    return run_burgers()


# Number of runs
N = 1

# Average time
time = timeit.timeit(time_function, number=N) / N

print(time)

# Save file
# file = open('execution_time', 'w')
# file.write('execution_time' + ' = ' + str(time) + ' seconds')
# file.close()

