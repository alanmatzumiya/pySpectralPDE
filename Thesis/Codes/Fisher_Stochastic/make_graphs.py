import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from main import run_fishers
from Graphics import Graph


def make():
    """
    This function constructs the graphs of the simulations
    and the calculation of the convergence

    Returns
    -------
    graph : object
    """
    
    xSpace, tim, simulation1, simulation2, norms, times = run_fishers()

    # Graphics creating
    Graph(xSpace, tim, simulation1, simulation2)

    # Graph convergence
    fig4, ax4 = plt.subplots(figsize=(10,10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(times, norms)

    plt.xlim(tim[0], tim[len(tim) - 1])
    plt.xlabel(r'\textit{time} (t)', fontsize=15)
    plt.ylabel(r'\textbf{$\| \Psi_{t}^{x} - \Psi_{t}^{y} \|_{\mathcal{L}(\mathcal{H}, \mu)^{2}}^{2}$}', fontsize=20)
    plt.title('Continuity with respect to the initial conditions',
              fontsize=16, color='black')
    ax4.grid()
    fig4.savefig('Graphics/Convergence')


make()

# plt.show()
