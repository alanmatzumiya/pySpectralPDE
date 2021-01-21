from pySpectralPDE import spectralSPDE
import numpy as np

def u0(x):
    return np.sin(np.pi * x)
    
t = np.linspace(0, 10, 512)
x = np.linspace(0, 1, 256)    
params = {"nu": 0.1, "N": int(5), "M": int(11), "x": x, "t": t}

sol = spectralSPDE.setup_solver(u0=u0, params=params)
sol.get_data
sol.plot
sol.u0_approx
sol.stability
