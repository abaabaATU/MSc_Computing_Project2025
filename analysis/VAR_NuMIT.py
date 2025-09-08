import numpy as np
from numpy.linalg import slogdet
from scipy.stats import wishart
from scipy.optimize import fsolve

from .VAR_utils import *
from .utils import Sigmoid
from .PhiID import PhiID_calculator


def VAR_from_MI(MI, p, S, A=None, V=None):
    """
    Generate a random VAR system with specified total mutual information (MI).

    Parameters:
        MI (float): Target total mutual information between sources and targets.
        p (int): Order of the VAR(p) model.
        S (int): Number of sources in the VAR system.
        A (numpy ndarray, optional): Starting coefficient matrix of the VAR system [shape (S, S, p)].

    Returns:
        A (numpy ndarray): Coefficient matrix of the VAR system with shape (S, S, p). 
        V (numpy ndarray): Noise covariance matrix (Wishart-distributed) of shape (S, S).
        success (int): Flag indicating the success (1) or failure (0) of the optimization process.
        sr (float): Optimized spectral radius for the system.
    """
    # Initialize variables
    x0 = [0, -1, 2, -2, 5, -5]  # Initial guesses for optimization

    # Generate random VAR coefficients
    if A is None:
        A = var_rand(S, p, np.random.rand())
        while np.any(np.abs(A) < 1e-18):
            A = var_rand(S, p, np.random.rand())
    else:
        if len(A.shape)==2: A = A[:,:,np.newaxis]
        assert A.shape[2] == p, "Model order and number of evolution matrices are not the same"
        assert A.shape[0] == S and A.shape[1]==S, "Sources and evolution matrix dimensions are not compatible"

    # Generate noise covariance matrix from Wishart distribution
    if V is None:
        V = wishart.rvs(df=S + 1, scale=np.eye(S))

    # Optimize to find the sectral radius `g` such that mutual information is `MI`
    def fun(x):
        x = Sigmoid(x, 1)
        B = specnorm(A, x)[0]
        try:
            G = var_to_autocov(B, V, 1)
            MI_value = (0.5 * np.linalg.slogdet(G[:, :, 0])[1] - 0.5 * np.linalg.slogdet(V)[1]) / np.log(2)
            if not np.isreal(MI_value):
                raise ValueError("Error in optimizing g: MI became complex!")
            return MI_value - MI
        except:
            return np.nan

    for x in x0:
        g, _, success, _ = fsolve(fun, x, full_output=1)
        if success: break
    else:
        print("Optimization failed")
        return None, None, success, None

    # Apply the optimized spectral radius and return the coefficients
    sr = Sigmoid(g, 1)
    A = specnorm(A, sr)[0]

    return A, V, success, sr


def PID_VAR_calculator(p, A, V=None, L1=None, L2=None, red_fun="MMI", as_dict=False, verbose=False):
    """
    Calculate the Partial Information Decomposition (PID) for a Vector Autoregressive (VAR) system.

    Parameters:
        p (int): Order of the VAR(p) model.
        A (numpy.ndarray): Coefficient matrix/matrices of the VAR system. 
                           Should have shape (L, L, p) or (L, L) if `p=1`.
        V (numpy.ndarray, optional): Noise covariance matrix of shape (L, L). Default is an identity matrix.
        L1 (int, optional): Dimension of the first source/target (S1, T1). If None, it is assumed to be half of `L`.
        L2 (int, optional): Dimension of the second source/target (S2, T2). If None, it is assumed to be half of `L`.
        red_fun (str, optional): Redundancy function to use. Supported options:
                                 - "MMI" (Minimal Mutual Information)
                                 - "CCS" (Common Change in Surprisal)
                                 Default is "MMI".
        as_dict (bool, optional): If True, returns the PID components as a dictionary. 
                                  Otherwise, returns them as a 1D numpy array. Default is False.

    Returns:
        PID (dict or numpy.ndarray): Partial Information Decomposition (PID) components
                                    if as_dict==False (default):
                                     - (4,) numpy.array: [R, U_X, U_Y, S]
                                    if as_dict==True:
                                     - Red: Redundancy
                                     - UnX: Unique information in X
                                     - UnY: Unique information in Y
                                     - Syn: Synergy
        Gamma (numpy.ndarray): Autocovariance matrices of the VAR model.
    """

    if (L1 is None) or (L2 is None):
        assert A.shape[0]%2==0, "Dimension of the VAR is odd, enter the dimensions of the two PID sources explicitly."
        L1 = A.shape[0] // 2
        L2 = L1
    L = L1 + L2
    if V is None: V = np.eye(L)
    assert p > 0, "Enter a valid model order"
    assert len(A.shape)==2 or len(A.shape)==3
    if len(A.shape)==2: A = A[:,:,np.newaxis]
    assert A.shape[2] == p, "Model order and number of evolution matrices are not the same"
    assert A.shape[0] == L and A.shape[1]==L, "Sources and evolution matrix dimensions are not compatible"

    # get autocovariance matrices
    Gamma = var_to_autocov(A, V, p, False)

    # get the full covariance of the system (future first, then past variables)
    full_cov = np.empty((L*(p+1),L*(p+1)))
    for s in range(p+1):
        for t in range(p+1):
            full_cov[s*L:(s+1)*L, t*L:(t+1)*L] = Gamma[:, :, abs(t-s)] if t > s else Gamma[:, :, abs(t-s)].T

    # get source and target covariances
    cov_S = full_cov[:-L, :-L]
    cov_T = Gamma[:,:,0]

    # calculate entropies
    H_T = 0.5 * slogdet(cov_T)[1]
    H_S = 0.5 * slogdet(cov_S)[1]
    H_TS = 0.5 * slogdet(full_cov)[1]

    # MI is in BIT
    MI = (H_T + H_S - H_TS) / np.log(2)


    # now calculate marginal entropies and MI
    sigma_T1S1 = np.zeros((L1*(p+1), L1*(p+1)))
    sigma_T2S2 = np.zeros((L2*(p+1), L2*(p+1)))

    for s in range(p+1):
        for t in range(p+1):
            sigma_T1S1[s*L1 : (s+1)*L1, t*L1 : (t+1)*L1] = Gamma[0:L1, 0:L1, abs(t-s)] if t>s else Gamma[0:L1, 0:L1, abs(t-s)].T 
            sigma_T2S2[s*L2 : (s+1)*L2, t*L2 : (t+1)*L2] = Gamma[L1:, L1:, abs(t-s)] if t>s else Gamma[L1:, L1:, abs(t-s)].T

    row_T2 = np.zeros((L2, L1*(p+1)))
    row_T1 = np.zeros((L1, L2*(p+1)))
    for l in range(p+1):
        row_T2[:L2, l*L1 : (l+1)*L1] = Gamma[L1:, :L1, l]
        row_T1[:L1, l*L2 : (l+1)*L2] = Gamma[:L1, L1:, l]

    addT2 = Gamma[L1:, L1:, 0]
    addT1 = Gamma[:L1, :L1, 0]

    sigma_TS1 = np.hstack((np.vstack((addT2, row_T2.T)),
                            np.vstack((row_T2, sigma_T1S1))))

    sigma_TS2 = np.hstack((np.vstack((addT1, row_T1.T)),
                            np.vstack((row_T1, sigma_T2S2))))

    sigma_S1 = sigma_T1S1[L1:, L1:]
    sigma_S2 = sigma_T2S2[L2:, L2:]

    # finally calculate entropies and MI 
    H_T = 0.5 * slogdet(cov_T)[1]
    H_S = 0.5 * slogdet(sigma_S1)[1]
    H_TS = 0.5 * slogdet(sigma_TS1)[1]
    MI_TX = (H_T + H_S - H_TS) / np.log(2)

    H_T = 0.5 * slogdet(cov_T)[1]
    H_S = 0.5 * slogdet(sigma_S2)[1]
    H_TS = 0.5 * slogdet(sigma_TS2)[1]
    MI_TY = (H_T + H_S - H_TS) / np.log(2)

    if red_fun == "MMI":
        R = min(MI_TX, MI_TY)
    elif red_fun == "CCS":
        c = MI - MI_TX - MI_TY
        signs = [np.sign(MI_TX), np.sign(MI_TY), np.sign(MI), np.sign(-c)]
        R = np.all(signs == signs[0], axis=0)*(-c)
    else:
        raise ValueError(f"{red_fun} redundancy function not supported!")
    
    U_X = MI_TX - R
    U_Y = MI_TY - R
    S = MI - U_Y - U_X - R

    if as_dict:
        PID = {
            "Red": R,
            "UnX": U_X,
            "UnY": U_Y,
            "Syn": S
        }
    else:
        PID = np.array([R, U_X, U_Y, S])

    return PID, Gamma, full_cov
    

def PhiID_VAR_calculator(p, A, V=None, L1=None, L2=None, red_fun="MMI", as_dict=False, verbose=False, **kwargs):
    """
    Calculate the Phi Information Decomposition (PhiID) for a Vector Autoregressive (VAR) system.

    Parameters:
        p (int): Order of the VAR(p) model. Currently only `p=1` is supported.
        A (numpy.ndarray): Coefficient matrix/matrices of the VAR system. 
                           Should have shape (L, L, 1) or (L, L).
        V (numpy.ndarray, optional): Noise covariance matrix of shape (L, L). Default is an identity matrix.
        L1 (int, optional): Dimension of the first source/target (S1, T1). If None, it is assumed to be half of `L`.
        L2 (int, optional): Dimension of the second source/target (S2, T2). If None, it is assumed to be half of `L`.
        red_fun (str, optional): Redundancy function to use. Supported options:
                                 - "MMI" (Minimal Mutual Information)
                                 - "CCS" (Common Change in Surprisal)
                                 - "Broja" (Broja)
                                 Default is "MMI".
        as_dict (bool, optional): If True, returns the PID components as a dictionary. 
                                  Otherwise, returns them as a 1D numpy array. Default is False.

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
        Gamma (numpy.ndarray): Autocovariance matrices of the VAR model.
    """
    if (L1 is None) or (L2 is None):
        assert A.shape[0]%2==0, "Dimension of the VAR is odd, enter the dimensions of the two PhiID sources explicitly."
        L1 = L2 = A.shape[0] // 2
    L = L1 + L2 
    
    if V is None: V = np.eye(L)
    assert p == 1, "Currently supports only VAR(1) systems"
    assert len(A.shape)==2 or len(A.shape)==3
    if len(A.shape)==2: A = A[:,:,np.newaxis]
    assert A.shape[2] == p, "Model order and number of evolution matrices are not the same"
    assert A.shape[0] == L and A.shape[1]==L, "Sources and evolution matrix dimensions are not compatible"

    # Solve the Lyapunov equation
    Gamma = var_to_autocov(A, V, p)
    
    # Full covariance matrix (past first, then future variables)
    Sigma_full = np.block([[Gamma[:,:,0], Gamma[:,:,1].T], [Gamma[:,:,1], Gamma[:,:,0]]])       

    atoms = PhiID_calculator(Sigma_full, L1, L2, red_fun, as_dict, verbose, **kwargs)

    return atoms, Gamma, Sigma_full



def VAR_nulls(MIs, kind, params, verbose=False, As=None, parallel=False, n_jobs=4, **kwargs):
    """
    Generates null models for a given set of mutual information (MI) values using Vector Autoregressive (VAR) processes. 

    Parameters
    ----------
    MIs : list or numpy.ndarray
        Array of mutual information (MI) values for which null models are generated.
        Each entry corresponds to a different MI value.
    kind : str
        Type of decomposition to be performed ("PID", PhiID")
    params : dict
        Configuration parameters for null model generation. Must include the following keys:
        - 'p' : list or numpy.ndarray
            List of model orders for each MI. If not provided, defaults to 1 for all MIs.
        - 'n_runs' : int
            Number of null model simulations to generate. Defaults to 1000.
        - 'S' : int
            Dimension of the VAR process. Defaults to 2.
            If odd, additional keys 'S1' and 'S2' must be provided.
        - 'red_fun' : str
            Redundancy function to use. Defaults to "MMI".
    verbose : bool, optional
        If True, prints progress and warnings. Defaults to False.
    As : list of numpy.ndarray, optional
        List of VAR coefficients for the null generation. If not provided, they are randomly sampled.
    parallel : bool, optional
        If True, generates null models in parallel using joblib. Defaults to False.
    n_jobs : int, optional
        Number of parallel jobs to run if `parallel` is True. Defaults to 4.
    **kwargs : additional keyword arguments

    Returns
    -------
    nulls : numpy.ndarray
        A 3D array of shape (n_atoms, len(MIs), params["n_runs"]) containing the null models
        for each MI value.
    """
    # input check
    if 'p' not in params:
        params['p'] = np.ones(len(MIs))
    if 'n_runs' not in params:
        params['n_runs'] = 1000
    if 'S' not in params:
        params['S'] = 2
    if params['S'] % 2 != 0:
        assert 'S1' in params and 'S2' in params, \
            "Dimension of the VAR is odd, enter the dimensions of the two PID sources explicitly."
    if 'red_fun' not in params:
        params['red_fun'] = "MMI"
    if not As:
        As = [None]*len(MIs)

    if kind=="PID":
        atom_calculator = PID_VAR_calculator
        n_atoms = 4
    elif kind=="PhiID":
        atom_calculator = PhiID_VAR_calculator
        n_atoms = 16
    else: 
        raise NotImplementedError
    
    def nulls_from_MI(mi, p, verbose):
        nerr_loc = 0
        mi_nulls = np.full((n_atoms, params["n_runs"]),np.nan)
        for n in range(params["n_runs"]):
            if n%(params["n_runs"]//10)==0 and verbose: print(f"{n/params['n_runs']*100}% null models completed!") 
            A,V,succ,_ = VAR_from_MI(mi, p, params["S"], As[np.random.randint(0,len(As))])
            if not succ:
                if verbose: print("Optimisation failed, skipping...")
                nerr_loc+=1
                continue
            mi_nulls[:,n] = atom_calculator(p, A, V, params["S"]//2, params["S"]//2, params["red_fun"], verbose=verbose, **kwargs)[0]
        return mi_nulls, nerr_loc

    nerr=0
    nulls = np.full((n_atoms,len(MIs),params["n_runs"]),np.nan)
    if parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(nulls_from_MI)(mi, params["p"][m], verbose) for m,mi in enumerate(MIs))
        for m, (nulls_res, nerr_res) in enumerate(results):
            nulls[:,m,:] = nulls_res
            nerr += nerr_res
        if verbose: print("All null models completed!")
    else:
        for m,mi in enumerate(MIs):
            if verbose: print(f"Generating null models for set {m+1} of {len(MIs)}.")
            nulls[:,m,:], mi_nerr = nulls_from_MI(mi, params["p"][m], verbose)
            nerr += mi_nerr
            if verbose: print(f"100% nulls models completed!\n")

    if verbose: print(f"{nerr} optimisations have failed!\n")
    return nulls