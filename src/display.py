import numpy as np
import matplotlib.pyplot as plt

def scatter(R):
    x = [i[0] for i in R]
    y = [i[1] for i in R]
    z = [i[2] for i in R]
    ax = plt.subplot(111, projection='3d')

    ax.scatter(x, y, z)

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def plot_momentum(mv):
    plt.figure()
    plt.title('Momentum')
    plt.ylabel('Momentum')
    plt.xlabel('Steps')
    plt.plot(mv)

def plot_E(E):
    plt.figure()
    plt.title('Energy')
    plt.ylabel('Energy')
    plt.xlabel('Steps')
    plt.plot(E)

def plot_T(T):
    plt.figure()
    plt.title('Temperature')
    plt.ylabel('Temperature')
    plt.xlabel('Steps')
    plt.plot(T)

def plot_gr(gr,nbins,L):
    dr = 0.5 * L / nbins
    x = np.arange(0,L/2,dr)

    plt.figure()
    plt.title('Pair Correlation Function')
    plt.ylabel('g(r)')
    plt.xlabel('r')

    plt.plot(x[:nbins],gr)

def plot_sk(kvecs,sk_list):
    kmags = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list)  # convert to numpy array if not already so

    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag = unique_kmags[iukmag]
        idx2avg = np.where(kmags == kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag

    # visualize
    plt.figure()
    plt.title('S(k)')
    plt.ylabel('S(k)')
    plt.xlabel('k')
    plt.plot(unique_kmags, unique_sk)

def plot_vv(vacf,t0,steps):
    plt.figure()
    plt.title('Velocity-velocity Correlation')
    plt.ylabel('Velocity-velocity Correlation')
    plt.xlabel('Steps')
    x = range(t0,steps)
    plt.plot(x,vacf)