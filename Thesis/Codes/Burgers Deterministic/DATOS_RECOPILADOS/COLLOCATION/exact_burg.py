def burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt ):
    import numpy as np
    from hermite_ek_compute import hermite_ek_compute

    qn = 8
    #
    #  Compute the rule.
    #
    qx, qw = hermite_ek_compute(qn)
    #
    #  Evaluate U(X,T) for later times.
    #
    vu = np.zeros([vxn, vtn])

    for vti in range(0, vtn):

        if (vt[vti] == 0.0):

            for i in range(0, vxn):
                vu[i, vti] = - np.sin(np.pi * vx[i])

        else:

            for vxi in range(0, vxn):

                top = 0.0
                bot = 0.0

                for qi in range(0, qn):
                    c = 2.0 * np.sqrt(nu * vt[vti])

                    top = top - qw[qi] * c * np.sin(np.pi * (vx[vxi] - c * qx[qi])) \
                          * np.exp(- np.cos(np.pi * (vx[vxi] - c * qx[qi])) \
                                   / (2.0 * np.pi * nu))

                    bot = bot + qw[qi] * c \
                          * np.exp(- np.cos(np.pi * (vx[vxi] - c * qx[qi])) \
                                   / (2.0 * np.pi * nu))

                    vu[vxi, vti] = top / bot

    return vu