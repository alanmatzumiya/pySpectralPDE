import numpy as np
from time import time
import os

from Fourier_Collocation import Simulation
#from ExactSol_Burgers import exact

xL = -60.0
xR = 60.0
tmax = 100

#M = [1500, 2000, 4500]
alphas = [0.025] 
Nj = 2**np.arange(12, 13) + 1
#dt = [0.00001]

#alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
#Nj = 2**np.arange(5, 13)
dt = [0.00001]


for i in range(0, 1):
    alpha = alphas[i]
    #Mi = M[i]
    for j in Nj:
        direction1 = 'Generated_Data/Simulation_Data/eps=' + str(alpha) + '/N=' + str(j)
        #direction2= 'Generated_Data/Exact_Solution/eps=' + str(alpha) + '/N=' + str(j)
        if not os.path.isdir(direction1):
            os.mkdir(direction1)
        #if not os.path.isdir(direction2):
        #   os.mkdir(direction2)
        for k in dt:
            exec_time = []
            ti = time()
            tdata, data, X = Simulation(j, xL, xR, tmax, k, alpha)
            t_exec = time() - ti
            if np.isnan(data).any() == False:
                direction3 = direction1 + '/dt=' + str(k)
                #direction4 = direction2 + '/dt=' + str(k)
                if not os.path.isdir(direction3):
                    os.mkdir(direction3)
                #if not os.path.isdir(direction4):
                #    os.mkdir(direction4)
                #exact_sol = exact(X, tdata, Mi, alpha)
                # simulation
                name = '/' + 'alpha=' + str(alpha) + '_N=' + str(j) + '_dt=' + str(k)
                np.save(direction3 + name + '_data', data)
                np.save(direction3 + name + '_space', X)
                np.save(direction3 + name + '_times', tdata)
                # exact
                #name = '/' + 'exact_' + 'alpha=' + str(alpha) + '_N=' + str(j) + '_dt=' + str(k)
                #np.save(direction4 + name + '_data', exact_sol)
                #np.save(direction4 + name + '_space', X)
                #np.save(direction4 + name + '_times', tdata)
                exec_time.append(t_exec)
                exec_time = np.array(exec_time)
                np.save(direction3 + name + '_exec_time', exec_time)
