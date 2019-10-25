import numpy as np

# The simulation will require most of the functions you have already
# implemented above. If it helps you debug, feel free to copy and
# paste the code here.
def my_distance(drij):
    """
    Compute length of displacement vector drij
    assume drij already accounts for PBC

    Args:
      drij (np.array) : vector(s) of length 3
    Returns:
      float: length (distance) of vector(s)
    """
    return np.linalg.norm(drij, axis=0)


def my_disp_in_box(drij, lbox):
    """
    Impose minimum image condition on displacement vector drij=ri-rj

    Args:
      drij (np.array): length-3 displacement vector ri-rj
      lbox (float): length of cubic cell
    Returns:
      np.array: drij under MIC
    """

    return drij - lbox * np.round(drij / lbox)


def my_pos_in_box(pos, lbox):
    """ wrap positions inside simulation box

    Args:
      pos (np.array): positions, shape (natom, ndim)
      lbox (float): box side length
    Returns:
      np.array: pos in box
    """

    return -lbox / 2 + (pos - lbox / 2) % lbox


def my_kinetic_energy(vel, mass):
    """ Calculate total kinetic energy.

    Args:
      vel (np.array): particle velocities, shape (natom, ndim)
      mass (float): particle mass
    Return:
      float: total kinetic energy
    """
    k = 0.0
    for i in vel:
        k += sum(0.5 * mass * i ** 2)

    return k


def my_potential_energy(rij):
    """ Calculate total potential energy.

    Args:
      rij (np.array): distance table, shape (natom, natom)
    Return:
      float: total potential energy
    """
    potential = 0.0
    for i in range(len(rij)):
        for j in range(i + 1, len(rij[0])):
            r = rij[i][j]
            potential += 4 * r ** (-6) * (r ** (-6) - 1)

    return potential


def my_force_on(i, pos, lbox):
    """
    Compute force on atom i

    Args:
      i (int): particle index
      pos (np.array) : particle positions, shape (natom, ndim)
      lbox (float): side length of cubic box
    Returns:
      np.array: force on atom i, a length-3 vector
    """
    Force = np.zeros(3)
    cur = pos[i]
    for atom in pos:
        if (atom == cur).all():
            continue
        r_ij = my_disp_in_box(cur - atom, lbox)
        r = my_distance(r_ij)
        Force += 24 * r ** (-8) * (2 * r ** (-6) - 1) * r_ij
    return Force


def get_distance_table(N, R, lbox):
    drij = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            disp = my_disp_in_box(R[i] - R[j], lbox)
            distance = my_distance(disp)
            drij[i][j] = distance
            drij[j][i] = distance
    return drij


def my_momentum(M, v):
    """ Calculate total momentum.

        Args:
          mass (float): particle mass
          vel (np.array): particle velocities, shape (natom, ndim)
        Return:
          float: total momentum
        """
    return sum(M*v)

def my_temperature(k,N):
    """ Calculate system temperature.

           Args:
             Ek (float): kinetic energy
             atoms (integer): number of atoms
           Return:
             float: temperature
           """
    return k/(3*N/2)

#-----------------------------Pair Correlation Function-----------------------------

def get_distance(dis_table):
    dists=[]
    for i in range(len(dis_table)):
        for j in range(i+1, len(dis_table)):
            dists.append(dis_table[i][j])

    return np.array(dists)

def my_histogram_distances(dists, nbins, dr):
    """ histogram an array of distances

    Histogram bin edges start at r=0 and are evenly spaced with
    sepration dr. (note: you need nbins+1 bin edges to enclosen nbins)

    Args:
      dists (np.array): 1d array of pair distances
      nbins (int): number of bins in histogram
      dr (float): bin size
    Return:
      np.array: counts, shape (nbins,), array of integer counts
    """
    counts = np.zeros(nbins, dtype=int)
    edges = np.arange(0, (nbins + 1) * dr, dr)

    for i in dists:
        for j in range(len(edges) - 1):
            if i <= edges[-1] and i > edges[j] and i <= edges[j + 1]:
                counts[j] += 1
    return counts


def my_pair_correlation(dists, natom, nbins, dr, lbox):
    """ Calculate the pair correlation function g(r).

    Histogram bin edges start at r=0 and are evenly spaced with
    sepration dr. (note: you need nbins+1 bin edges to enclosen nbins)
    Normalization for bin i uses bin center r=edge[i] + 0.5*dr in
    the formula \Omega(r).

    Args:
      dists (np.array): 1d array of pair distances
      natom (int): number of atoms
      nbins (int): number of bins to histogram
      dr (float): size of bins
      lbox (float): side length of cubic box
    Return:
      np.array: gr, shape (nbins,), pair correlation function as a normalized histogram
    """
    gr = np.zeros(nbins)
    counts = my_histogram_distances(dists, nbins, dr)

    V = lbox ** 3

    for i in range(len(counts)):
        vbin = (4 * 3.1415926 / 3) * (((i + 1) * dr) ** 3 - (i * dr) ** 3)
        omega = V / (vbin * natom * (natom - 1) / 2)
        gr[i] = omega * counts[i]

    return gr

#-----------------------------Structure Factor-----------------------------

def my_legal_kvecs(maxn, lbox):
    """ Calculate k vectors commensurate with a cubic box.

    Consider only k vectors in the all-positive octant of reciprocal space.

    Args:
      maxn : maximum value for nx, ny, nz; maxn+1 is number of k-points along each axis
      lbox : side length of cubic cell

    Return:
      np.array: kvecs, shape (nk, ndim), array of k-vectors
    """

    N = (maxn + 1) ** 3
    kvecs = np.zeros((N, 3))
    n = 0
    for x in range(maxn + 1):
        for y in range(maxn + 1):
            for z in range(maxn + 1):
                if n < N:
                    kvecs[n, 0] = 2 * 3.1415926 / lbox * x
                    kvecs[n, 1] = 2 * 3.1415926 / lbox * y
                    kvecs[n, 2] = 2 * 3.1415926 / lbox * z
                n += 1

    return kvecs

def my_calc_rhok(kvecs, pos):
  """ Calculate the fourier transform of particle density.

  Args:
    kvecs (np.array): array of k-vectors, shape (nk, ndim)
    pos (np.array): particle positions, shape (natom, ndim), or (nstep, natom, ndim)
  Return:
    np.array: rho, shape (nk,), fourier transformed density
  """
  nk = len(kvecs)
  rhok = np.zeros(nk, dtype=complex)
  for r in pos:
      rhok += np.e**(-1j * kvecs.dot(r))
  return rhok

def my_calc_sk(kvecs, pos):
  """ Calculate the structure factor S(k).

  Args:
    kvecs (np.array): array of k-vectors, shape (nk, ndim)
    pos (np.array): particle positions, shape (natom, ndim), or (nstep, natom, ndim)
  Return:
    np.array: sk, shape (nk,), structure factor at each k-vector
  """
  sk = my_calc_rhok(kvecs, pos) * my_calc_rhok(-kvecs, pos) / len(pos)
  return sk

#-----------------------Velocity-velocity correlation--------------------------

def my_calc_vacf(all_vel, t0, steps):
    """ Calculate the vacf.

    Args:
      all_vel (np.array): shape (nstep, natom, ndim), all particle velocities
       from an MD trajectory. For example all_vel[t, i] is the velocity
       vector of particle i at time t.
      t (int): time index at which the velocity velocity correlation is
       calculated.
    Returns:
      float: the vv correlation calculated at time index t.
    """
    vacf = np.zeros(steps-t0)
    for t in range(t0,steps):
        for i in range(len(all_vel[0])):
            vacf[t-t0] += all_vel[t0,i].dot(all_vel[t,i])
        vacf[t-t0] /= len(all_vel[0])
    return vacf

#--------------------------Diffusion constant---------------------------------

def my_diffusion_constant(vacf,h):
    """ Calculate the diffusion constant from the
    velocity-velocity auto-correlation function (vacf).

    Args:
      vacf (np.array): shape (nt,), vacf sampled at
       nt time steps from t=0 to nt*dt. dt is set to 0.032.
    Returns:
      float: the diffusion constant calculated from the vacf.
    """
    D = 0
    for i in range(len(vacf)-1):
        D += (vacf[i] + vacf[i+1]) * h / 2
    D /= 3
    return D