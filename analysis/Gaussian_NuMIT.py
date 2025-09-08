import numpy as np
import os, sys
from scipy.stats import wishart
from scipy.optimize import fsolve
from .utils import h, RedMMI, RedCCS
from .PhiID import PhiID_calculator

# TODO: implement Broja PID (use gpid package)
# TODO: PhiID gaussian fitness procedure

def Gauss_from_MI(MI, S=2, T=1, A=None, Sigma_s=None, Sigma_u=None):
    """
    Generate a random Gaussian system with a specified total Mutual Information (MI) between sources and targets.

    Parameters:
    - MI: Desired total mutual information between sources and targets (scalar).
    - S: Number of sources in the system (default is 2).
    - T: Number of targets in the system (default is 1).
    - A: [T, S] matrix of linear coefficients relating sources to targets (optional).
    - Sigma_s: [S, S] covariance matrix of the sources (optional).
    - Sigma_u: [T, T] conditional covariance matrix of the targets given the sources (optional).

    Returns:
    - Sigma: [S+T, S+T] covariance matrix of the full system.
    - success: Scalar flag indicating success (1) or failure (0) of the mutual information optimization.
    - A: [T, S] matrix of linear coefficients used in the system.
    - g: Scalar scaling factor applied to the conditional covariance matrix.
    """
    x0 = [0, -1, 2, -2, 5, -5]
    success = 0

    if Sigma_s is not None:
        Sigma_s = wishart.rvs(df=S+1, scale=np.eye(S))
    if A is not None:
        A = np.random.normal(0, 1, size=(T, S))
    if Sigma_u is not None:
        Sigma_u = np.atleast_2d(wishart.rvs(df=T+1, scale=np.eye(T)))

    def fun(x):
        # NB: mutual information is in BITS
        return np.linalg.det(np.eye(T) + x * np.linalg.solve(Sigma_u, A @ Sigma_s @ A.T)) - 2**(2 * MI)

    alpha = None
    for x in x0:
        alpha, _, success, _ = fsolve(fun, x, full_output=1)
        if success: break
    else: 
        print("Optimization failed")
        return None, None, success, None
    
    g = 1 / alpha
    Sigma_t = A @ Sigma_s @ A.T + g * Sigma_u
    Sigma_cross = A @ Sigma_s
    Sigma = np.block([[Sigma_s, Sigma_cross.T], [Sigma_cross, Sigma_t]])

    return Sigma, success, A, g


def PID_Gauss_calculator(Sigma, S=2, T=1, S1=None, S2=None, red_fun="MMI", as_dict=False, verbose=False):
    """
    Compute Partial Information Decomposition (PID) atoms for a Gaussian system T = A*S + U.

    Parameters:
    - Sigma: [S+T, S+T] covariance matrix of the system.
    - S: Number of sources in the system.
    - T: Number of targets in the system.

    Returns:
    - UnX: Unique information from the first source.
    - UnY: Unique information from the second source.
    - Red: Redundancy.
    - Syn: Synergy.
    - MI_S1: Marginal mutual information between the first source and the target.
    - MI_S2: Marginal mutual information between the second source and the target.
    - MI: Total mutual information between the sources and the target.
    """
    if Sigma.shape[0] != S + T:
        raise ValueError("Mismatch between number of sources/targets and covariance dimensions.")
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Input covariance matrix not positive definite!")
    if S % 2 != 0:
        assert S1 and S2, \
            "Dimension of S is odd, enter the dimensions of the two PID sources explicitly."
    if not S1 or not S2:
        S1 = S//2 
        S2 = S//2

    # Extract relevant covariance submatrices
    Sigma_t = Sigma[-T:, -T:]
    Sigma_s = Sigma[:S, :S]
    Sigma_cross = Sigma[:S, S:].T

    # Total mutual information
    # NB mutual information is in BITS
    MI = (h(Sigma_t) + h(Sigma_s) - h(Sigma)) / np.log(2)

    # Mutual information between the first source and the target
    Sigma_TS1 = np.block([
        [Sigma_t, Sigma_cross[:, :S1]],
        [Sigma_cross[:, :S1].T, Sigma_s[:S1, :S1]]
    ])
    MI_S1 = (h(Sigma_t) + h(Sigma_s[:S1, :S1]) - h(Sigma_TS1)) / np.log(2)

    # Mutual information between the second source and the target
    Sigma_TS2 = np.block([
        [Sigma_t, Sigma_cross[:, S1:]],
        [Sigma_cross[:, S1:].T, Sigma_s[S1:, S1:]]
    ])
    MI_S2 = (h(Sigma_t) + h(Sigma_s[S1:, S1:]) - h(Sigma_TS2)) / np.log(2)

    # PID atoms using the MMI definition
    if red_fun=="MMI":
        R = RedMMI(MI_S1, MI_S2)
    elif red_fun=="CCS":
        R = RedCCS(MI_S1, MI_S2, MI)

    UnX = MI_S1 - R
    UnY = MI_S2 - R
    S = MI - UnX - UnY - R

    if as_dict:
        PID = {
            "Red": R,
            "UnX": UnX,
            "UnY": UnY,
            "Syn": S
        }
    else:
        PID = np.array([R, UnX, UnY, S])

    return PID


def PhiID_Gauss_calculator(Sigma, S=2, T=None, S1=None, S2=None, red_fun="MMI", as_dict=False, verbose=False, **kwargs):
    """
    Compute Integrated Information Decomposition (PhiID) atoms for a Gaussian system T = A*S + U.

    Parameters:
    - Sigma: [2*(S1+S2), 2*(S1+S2)] covariance matrix of the system.
    - S: Number of sources in the system.
    - T: Number of targets in the system (equal to the number of sources - kept for compatibility).
    - S1: Number of sources in the first PhiID source.
    - S2: Number of sources in the second PhiID source.
    - red_fun: Redundancy function to use (default is "MMI").
    - as_dict: If True, returns the atoms as a dictionary (default is False).
    - verbose: If True, prints progress and warnings (default is False).
    
    Returns:
        atoms (dict or numpy.ndarray): PhiID components ordered as:
                                        if as_dict==False (default):
                                         (16,) numpy.array: [rtr, rtx, rty, rts, xtr, xtx, 
                                            xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts]
                                        if as_dict==True:
                                         dict with keys:
                                         - rtr, rta, rtb, rts, 
                                           xtr, xta, xtb, xts, 
                                           ytr, yta, ytb, yts, 
                                           str, sta, stb, sts
    """
    
    if Sigma.shape[0] != 2*S:
        raise ValueError("Mismatch between number of sources and covariance dimensions.")
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Input covariance matrix not positive definite!")
    if S % 2 != 0:
        assert S1 and S2, \
            "Dimension of S is odd, enter the dimensions of the two PhiID sources explicitly."
    if not S1 or not S2:
        S1 = S//2 
        S2 = S//2

    atoms = PhiID_calculator(Sigma, S1, S2, red_fun, as_dict, verbose, **kwargs)

    return atoms, Sigma


def Gauss_nulls(MIs, kind, params, verbose=False):
    """
    Generates null models for a given set of mutual information (MI) values using Gaussian processes. 

    Parameters
    ----------
    MIs : list or numpy.ndarray
        Array of mutual information (MI) values for which null models are generated.
        Each entry corresponds to a different MI value.
    kind : str
        Type of decomposition to be performed ("PID", PhiID")
    params : dict
        Configuration parameters for null model generation. Must include the following keys:
        - 'n_runs' : int
            Number of null model simulations to generate. Defaults to 1000.
        - 'S' : int
            Dimension of the Gaussian sources. Defaults to 2.
            If odd, additional keys 'S1' and 'S2' must be provided.
        - 'T' : int
            Dimension of the Gaussian target. Defaults to 1.
        - 'red_fun' : str
            Redundancy function to use. Defaults to "MMI".

    verbose : bool, optional
        If True, prints progress and warnings. Defaults to False.

    Returns
    -------
    nulls : numpy.ndarray
        A 3D array of shape (n_atoms, len(MIs), params["n_runs"]) containing the null models
        for each MI value.
    """

    # input check
    if 'n_runs' not in params:
        params['n_runs'] = 1000
    if 'S' not in params:
        params['S'] = 2
    if 'T' not in params:
        if kind=="PID":
            params['T'] = 1
        elif kind=="PhiID":
            params['T'] = params['S']
    if params['S'] % 2 != 0:
        assert 'S1' in params and 'S2' in params, \
            "Dimension of S is odd, enter the dimensions of the two PID sources explicitly."
    if 'red_fun' not in params:
        params['red_fun'] = "MMI"
    

    if kind=="PID":
        atom_calculator = PID_Gauss_calculator
        n_atoms = 4
    elif kind=="PhiID":
        atom_calculator = PhiID_Gauss_calculator
        n_atoms = 16
    else: 
        raise NotImplementedError

    nerr=0
    nulls = np.full((n_atoms,len(MIs),params["n_runs"]),np.nan)
    for m,mi in enumerate(MIs):
        if verbose: print(f"Generating null models for set {m+1} of {len(MIs)}.")
        for n in range(params["n_runs"]):
            if n%(params["n_runs"]/10)==0 and verbose: print(f"{n/params['n_runs']*100}% null models completed!")
            Sigma,succ,_,_ = Gauss_from_MI(mi, params["S"], params["T"])
            if succ==False:
                if verbose: print("Optimisation failed, skipping...")
                nerr+=1
                continue
            nulls[:,m,n] = atom_calculator(Sigma, params["S"], T=params["T"], red_fun=params["red_fun"])[0]
        if verbose: print(f"100% nulls models completed!\n")

    if verbose: print(f"{nerr} optimisations have failed!\n")
    return nulls


if __name__ == "__main__":
    MI = 4.45455
    Sigma, success, A, g = Gauss_from_MI(MI, S=2, T=2)

    # I_check = h(Sigma_t) + h(Sigma_s) - h(Sigma)
    # print(I_check)
    # if abs(MI - I_check) > 1e-6:
    #     success = 1
    #     print("Error in the optimization of the MI.")

    # Sigma = np.array([[ 1.02742114, -1.33018592,  0.50852778],
    #     [-1.33018592,  1.74101346, -0.67543034],
    #     [ 0.50852778, -0.67543034,  0.26715794]])

    # X = np.random.rand(4,4)
    # Sigma = X@X.T

    print(Sigma.shape, A.shape)
    print(f"PID atoms: {PID_Gauss_calculator(Sigma, S=2, T=2, red_fun='MMI')}")
    print(f"PhiID atom: {PhiID_Gauss_calculator(Sigma, red_fun='Broja')})")