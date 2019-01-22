import timeit
from main import run_fishers


def time_function():

    return run_fishers()


# Number of runs
N = 10

# Average time
time = timeit.timeit(time_function, number=N) / N

# Save file
file = open('execution_time', 'w')
file.write('execution_time' + ' = ' + str(time) + ' seconds')
file.close()

