import matplotlib.pyplot as plt
import random as rd

from source import display, initial as init, verlet as verl
from source.properties import *
from source.input import *


# Main Loop.
# ------------------------------------------------------------------------

# R, V, and A are the position, velocity, and acceleration of the atoms
# respectively. nR, nV, and nA are the next positions, velocities, etc.
# There are missing pieces in the code below that you will need to fill in.
# These are marked below with comments:

def simulate():
    """Initialize and run a simulation in a Ncube**3 box, for steps"""
    N = Ncube ** 3
    R = init.InitPositionCubic(Ncube, L)
    V = init.InitVelocity(N, T0, M)
    all_vec = np.zeros((steps, N, 3))
    A = np.zeros((N, 3))

    E = np.zeros(steps)
    T = np.zeros(steps)
    mv = np.zeros((steps,3))
    print("steps\ttemperature\tkinetic\tpotential\tenergy\t")

    for t in range(0, steps):
        # -----------------------Measuring Physical Properties----------------------------

        ## calculate momentum
        mv[t] = my_momentum(M, V)

        ## calculate kinetic energy contribution
        k = my_kinetic_energy(V, M)

        ## calculate temperature
        T[t] = my_temperature(k, N)

        ## calculate distance table
        drij = get_distance_table(N, R, L)

        ## calculate velocity table
        all_vec[t] = V

        ## calculate potential energy contribution
        p = my_potential_energy(drij)

        ## calculate total energy
        E[t] = k + p

        ## calculate forces; should be a function that returns an N x 3 array
        F = np.array([my_force_on(i, R, L) for i in range(N)])
        A = F / M

        # -----------------------Anderson Thermostat----------------------
        if anderson == True:
            sigma = (Ta / M) ** 0.5
            mean = 0
            for i in V:
                if np.random.random() < eta * h:
                    i[0] = rd.gauss(mean, sigma)
                    i[1] = rd.gauss(mean, sigma)
                    i[2] = rd.gauss(mean, sigma)

        # -----------------------Propagation----------------------------

        nR = verl.VerletNextR(R, V, A, h)
        nR = my_pos_in_box(nR, L)  ## from PrairieLearn HW

        ## calculate forces with new positions nR
        nF = np.array([my_force_on(i, nR, L) for i in range(N)])
        nA = nF / M
        nV = verl.VerletNextV(V, A, nA, h)

        # update positions:
        R, V = nR, nV

        # ------------------------Output-------------------------------------

        print('%d\t%.3f\t%.5f\t%.5f\t%.5f\t' % (t, T[t], k, p, E[t]))

    ## calculate pair correlation function
    nbins = 100
    dists = get_distance(drij)
    gr = my_pair_correlation(dists, N, nbins, 0.5 * L / nbins, L)

    ## calculate structure factor
    kvecs = my_legal_kvecs(5, L)
    sk = my_calc_sk(kvecs, R)

    ## calculate Velocity-velocity correlation
    t0 = 50
    vacf = my_calc_vacf(all_vec, t0, steps)

    ## calculate Diffusion constant
    D = my_diffusion_constant(vacf, h)
    print("Diffusion constant: " + str(D))

    # -----------------------------Display---------------------------------
    display.plot_E(E)
    display.plot_T(T)
    display.plot_momentum(mv)
    display.plot_gr(gr, nbins, L)
    display.plot_sk(kvecs, sk)
    display.plot_vv(vacf, t0, steps)
    plt.show()

    return E


if __name__ == '__main__':
    E = simulate()
