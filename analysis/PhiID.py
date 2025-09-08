import numpy as np
import sys, os
from .utils import h, RedMMI, doubleRedMMI, RedCCS, doubleRedCCS, arrayfy


def PhiID_calculator(Sigma, L1=None, L2=None, red_fun="MMI", as_dict=False, verbose=False, **kwargs):
    """
    Wrapper to calculate the PhiID lattice of a bivariate system with two sources and two targets.

    Parameters:
        Sigma (numpy.ndarray):      Covariance matrix of the system.
        L1 (int):                   Dimension of the first source.
        L2 (int):                   Dimension of the second source.
        red_fun (str):              Redundancy function to use.
        as_dict (bool, optional):   If True, return the lattice as a dictionary.
        verbose (bool, optional):   If True, print the atoms and optimisation details (only for Broja).

    Returns:
        PhiID atoms.
    """

    assert Sigma.shape[0] == Sigma.shape[1], "Covariance matrix must be square."
    if L1 is None or L2 is None:
        assert (Sigma.shape[0]//2)%2==0, "Dimension of the source covariance is odd, enter the dimensions of the two PhiID sources explicitly."
        L1 = L2 = Sigma.shape[0] // 4
    assert L1+L2 == Sigma.shape[0]//2, "Mismatch between legths of sources and dimension of source covariance."

    if red_fun == "Broja": 
        sys.path.append(os.path.join(os.getcwd(), "..", "Broja_PhiID"))
        from phiid import broja_phiid

        atoms = broja_phiid(Sigma, nx=L1, ny=L2, verbose=verbose, **kwargs)
        if not as_dict:
            atoms = arrayfy(atoms).flatten()
        return atoms
    
    elif red_fun == "MMI":
        Red = RedMMI
        doubleRed = doubleRedMMI
    elif red_fun == "CCS":
        Red = RedCCS
        doubleRed = doubleRedCCS
    else:
        raise ValueError("Redundancy function not supported! Choose between MMI, CCS, and Broja.")
    
    return PhiID(Sigma, L1, L2, Red, doubleRed, as_dict=as_dict)



def PhiID(Sigma, L1, L2, Red, doubleRed, as_dict=False):
    """
    Calculate the PhiID lattice of a bivariate system with two sources and two targets.

    Parameters:
        Sigma (numpy.ndarray):      Covariance matrix of the system.
        L1 (int):                   Dimension of the first source.
        L2 (int):                   Dimension of the second source.
        Red (function):             Function to calculate redundancy.
        doubleRed (function):       Function to calculate double redundancy.
        as_dict (bool, optional):   If True, return the lattice as a dictionary.

    Returns:
        atoms (numpy.ndarray or dict):  PhiID atoms.
    """
    import logging

    # Configure logging
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    assert Sigma.shape[0] == Sigma.shape[1], "Covariance matrix must be square."
    assert L1 > 0 and L2 > 0, "Length of sources/targets must be greater than zero."
    assert isinstance(L1, int) and isinstance(L2, int), "Length of sources/targets must be an integer."
    # eigenvalues = np.linalg.eigvals(Sigma)
    # if np.any(eigenvalues <= 0):
    #     negative_vals = eigenvalues[eigenvalues <= 0]
    #     logging.warning(f"Non-positive eigenvalues detected: {negative_vals}. ")
    
    L = L1 + L2

    # Define index ranges
    idx_x = list(range(L1))  # First source (past)
    idx_y = list(range(L1, L))  # Second source (past)
    idx_a = list(range(L, L + L1))  # First source (future)
    idx_b = list(range(L + L1, 2*L))  # Second source (future)

    H_xyab = h(Sigma)
    H_xy = h(Sigma[np.ix_(idx_x + idx_y, idx_x + idx_y)])
    H_ab = h(Sigma[np.ix_(idx_a + idx_b, idx_a + idx_b)])

    # Individual sources and targets
    H_x = h(Sigma[np.ix_(idx_x, idx_x)])
    H_a = h(Sigma[np.ix_(idx_a, idx_a)]) # H_a = H_x for stationary processes!
    H_y = h(Sigma[np.ix_(idx_y, idx_y)])
    H_b = h(Sigma[np.ix_(idx_b, idx_b)]) # H_b = H_y for stationary processes!

    # Combinations of sources and targets
    H_xab = h(Sigma[np.ix_(idx_x + idx_a + idx_b, idx_x + idx_a + idx_b)])
    H_yab = h(Sigma[np.ix_(idx_y + idx_a + idx_b, idx_y + idx_a + idx_b)])
    H_xya = h(Sigma[np.ix_(idx_x + idx_y + idx_a, idx_x + idx_y + idx_a)])
    H_xyb = h(Sigma[np.ix_(idx_x + idx_y + idx_b, idx_x + idx_y + idx_b)])

    # Mixed pairs of sources and targets
    H_xa = h(Sigma[np.ix_(idx_x + idx_a, idx_x + idx_a)])
    H_xb = h(Sigma[np.ix_(idx_x + idx_b, idx_x + idx_b)])
    H_ya = h(Sigma[np.ix_(idx_y + idx_a, idx_y + idx_a)])
    H_yb = h(Sigma[np.ix_(idx_y + idx_b, idx_y + idx_b)])

    # Calculate mutual information
    MI_xytab = (H_xy + H_ab - H_xyab) / np.log(2)
    MI_xta = (H_x + H_a - H_xa) / np.log(2)
    MI_yta = (H_y + H_a - H_ya) / np.log(2)
    MI_xtb = (H_x + H_b - H_xb) / np.log(2)
    MI_ytb = (H_y + H_b - H_yb) / np.log(2)

    MI_xyta = (H_xy + H_a - H_xya) / np.log(2)
    MI_xytb = (H_xy + H_b - H_xyb) / np.log(2)
    MI_xtab = (H_x + H_ab - H_xab) / np.log(2)
    MI_ytab = (H_y + H_ab - H_yab) / np.log(2)
    
    Rxyta = Red(MI_xta, MI_yta, MI_xyta)
    Rxytb = Red(MI_xtb, MI_ytb, MI_xytb)
    Rabtx = Red(MI_xta, MI_xtb, MI_xtab)
    Rabty = Red(MI_yta, MI_ytb, MI_ytab)
    Rxytab = Red(MI_xtab, MI_ytab, MI_xytab)
    Rabtxy = Red(MI_xyta, MI_xytb, MI_xytab)

    doubleR = doubleRed(MI_xta, MI_xtb, MI_yta, MI_ytb,
                        MI_xtab, MI_ytab, MI_xyta, MI_xytb,
                        MI_xytab, Rxyta, Rxytb, Rabtx, Rabty, Rxytab, Rabtxy)

    MIs = [doubleR, Rxyta, Rxytb, Rxytab, Rabtx, Rabty, Rabtxy,
            MI_xta, MI_xtb, MI_yta, MI_ytb, MI_xyta, MI_xytb, MI_xtab, MI_ytab, MI_xytab]

    # Define the matrix Mat
    Mat = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # doubleR
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxyta
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytb
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytab
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rabtx
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Rabty
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Rabtxy
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # MI_xta
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # MI_xtb
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # MI_yta
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # MI_ytb
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # MI_xyta
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # MI_xytb
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # MI_xtab
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # MI_ytab
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # MI_xytab
    ])

    lattice = np.linalg.solve(Mat, np.array(MIs).T)

    # If required, sort the results as dict
    if as_dict:
        atoms = {
            "rtr": lattice[0],
            "rta": lattice[1],
            "rtb": lattice[2],
            "rts": lattice[3],
            "xtr": lattice[4],
            "xta": lattice[5],
            "xtb": lattice[6],
            "xts": lattice[7],
            "ytr": lattice[8],
            "yta": lattice[9],
            "ytb": lattice[10],
            "yts": lattice[11],
            "str": lattice[12],
            "sta": lattice[13],
            "stb": lattice[14],
            "sts": lattice[15]
        }
    else: 
        atoms = lattice

    return atoms