Numerical approximation of the Burger's Stochastic equation.
====================================================
Numerical solution of stochastic Burger's equation on 1D using Spectral method based on the spectral decomposition of the Ornstein-Uhlenbeck semigroup associated to the Kolmogorov equation and compute the norm between two solutions with different initial conditions for a fixed point in real space.
----------------------------------------------------------------------------------------------------------

### Simulation Considerations.

Initial Condition (1).

<a href="https://www.codecogs.com/eqnedit.php?latex=x(\xi)&space;=&space;\sin(\pi&space;\xi),&space;\hspace{2mm}&space;\xi&space;\in&space;[0,&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x(\xi)&space;=&space;\sin(\pi&space;\xi),&space;\hspace{2mm}&space;\xi&space;\in&space;[0,&space;1]" title="x(\xi) = \sin(\pi \xi), \hspace{2mm} \xi \in [0, 1]" /></a>

Initial Condition (2).

Approximation with Chebyshev Polynomials

<a href="https://www.codecogs.com/eqnedit.php?latex=x(\xi)&space;\approx&space;y(\xi)&space;=&space;\sum^{3}_{n=0}&space;a_n&space;T_n&space;(\xi),&space;\hspace{2mm}&space;\xi&space;\in&space;[0,&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x(\xi)&space;\approx&space;y(\xi)&space;=&space;\sum^{3}_{n=0}&space;a_n&space;T_n&space;(\xi),&space;\hspace{2mm}&space;\xi&space;\in&space;[0,&space;1]" title="x(\xi) \approx y(\xi) = \sum^{3}_{n=0} a_n T_n (\xi), \hspace{2mm} \xi \in [0, 1]" /></a>

### Parameters of the simulation for (1) and (2).

<a href="https://www.codecogs.com/eqnedit.php?latex=\nu&space;=&space;0.01,&space;\hspace{2mm}&space;\math{N}&space;=&space;5,&space;\hspace{2mm},&space;\math{M}&space;=&space;11,&space;\hspace{2mm}&space;\math{t}&space;\in&space;[0,&space;10]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nu&space;=&space;0.01,&space;\hspace{2mm}&space;\math{N}&space;=&space;5,&space;\hspace{2mm},&space;\math{M}&space;=&space;11,&space;\hspace{2mm}&space;\math{t}&space;\in&space;[0,&space;10]" title="\nu = 0.01, \hspace{2mm} \math{N} = 5, \hspace{2mm}, \math{M} = 11, \hspace{2mm} \math{t} \in [0, 10]" /></a>

with a discretization of 2048 points in space and 1026 points in time.

The Python code 'main.py' returns the discretization of space and time; also returns the numerical approximations for the initial conditions (1) and (2), and the distances between them using the norm established in the paper.

The folder called 'Generated_Data' contains the data produced in the simulation given by the code 'main.py'. For more information about the data read 'README.md' that is contained in this folder.

The folder called 'Graphics' contains the figures produced by the Python codes 'make_graphs.py' and 'Graphics.py'

