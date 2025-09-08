import numpy as np
import os
from .VAR_NuMIT import PID_VAR_calculator, PhiID_VAR_calculator
from .VAR_fitness import fit_var
from .utils import saveData
import warnings


def VAR_analysis(data, subject_info, par={}, on_file=False, path="Results", verbose=False, overwrite=False):
    """
    Performs VAR analysis on given multivariate time series data.

    Parameters:
    -----------
    data : ndarray
        Multivariate time series data with shape (n_channels, n_timepoints, n_trials).
    subject_info : dict
        Information about the subject, must include the sampling frequency 'fs'.
    par : dict, optional
        Parameters for the analysis, which include:
        - 'channels': Number of channels to analyze (default: 2)
        - 'trials': Number of trials to analyze (default: 30)
        - 'runs': Number of VAR model runs (default: 100)
        - 'red_fun': Redundancy function ("MMI", "CCS" for PID, "MMI", "CCS", "Broja" for PhiID. default: "MMI")
        - 'method': Method to compute information decomposition atoms ('PID' or 'PhiID', default: "PID")
        - 'mmorder': Model order (default: 30 for PID, 1 for PhiID)
    on_file : bool, optional
        Whether to save the results to a file (default: False).
    path : str, optional
        Directory to save the results, only useful if `on_file=True` (default: "Results").
    verbose : bool, optional
        Whether to print progress messages (default: False).
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, only useful if `on_file=True` (default: False).

    Returns:
    --------
    results : dict
        A dictionary containing the analysis results, which include:
        - 'subject_info': Information about the subject.
        - 'parameters': The input parameters used for the analysis.
        - 'atoms': Computed atoms for each VAR model run.
    """

    # Default parameters
    par.setdefault('channels', 2)
    par.setdefault('trials', 30)
    par.setdefault('runs', 100)
    par.setdefault('red_fun', "MMI")
    par.setdefault('method', "PID")

    if par['method'] not in ["PID", "PhiID"]:
        raise ValueError("Invalid method. Choose PID or PhiID.")
    
    n_chans, n_timepoints, n_trials = data.shape

    # Adjust trials if needed
    par['trials'] = min(par['trials'], n_trials)

    results = {
        "subject_info": subject_info,
        "parameters": par
    }

    if par["method"]=="PID":
        par.setdefault('mmorder', 30)
        atoms = np.zeros(shape=(4, par['runs']))
        atoms_calculator = PID_VAR_calculator
    elif par["method"]=="PhiID":
        par.setdefault('mmorder', 1)
        if par['mmorder']!=1:
            warnings.warn("PhiID is only defined for VAR(1) models. Setting p=1.")
            par['mmorder']=1
        atoms = np.zeros(shape=(16, par['runs']))
        atoms_calculator = PhiID_VAR_calculator
    else:
        raise ValueError("Invalid method. Choose PID or PhiID.")

    for N in range(par['runs']):
        if verbose and N*100/par["runs"]%10==0:
            print(f"Done {N*100/par['runs']}%")

        # Random selection of channels and trials
        channels = np.random.choice(n_chans, par['channels'], replace=False)
        trials = np.random.choice(n_trials, par['trials'], replace=False)
        # ts = data[channels,:,:][:,:,trials]

        # # Fit VAR model
        # A, V, p, _ = fit_var(ts, maxp=par['mmorder'])

        # Testing steps
        ts = data[channels,:,:][:,:,trials]
        n, T, m = ts.shape
        # print(f"Debug - n: {n}, T: {T}, m: {m}")

        # checking if maxp is reasonable
        maxp = par.get('mmorder', int(12 * (T/100.)**(1./4)))
        maxp = min(maxp, T - 1)

        # retry
        A, V, p, _ = fit_var(ts, maxp=maxp)

        # Compute atoms
        atoms[:,N] = atoms_calculator(p, A, V, par['channels']//2, par['channels']//2, par['red_fun'], verbose=verbose)[0]
    if verbose: print(f"Done 100%")

    # Save results
    results["atoms"] = atoms
    if on_file:
        if path[-1]=="/":
            path = path[:-1]
            
        # Output directory: if it does not exist, create it
        dir = f"{path}/{par['method']}/{subject_info['drug']}/s{subject_info['name']}_{subject_info['condition']}/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        name = f"{par['red_fun']}_c{par['channels']}_results.pickle"
        fullpath = dir+name
        # Give a warning if the file is being overwritten
        if os.path.exists(fullpath):
            if overwrite:
                warnings.warn(f"File {fullpath} already exists. Overwriting...")
                saveData(results, fullpath)
            else:
                warnings.warn(f"File {fullpath} already exists. Skipping...")
        else:
            saveData(results, fullpath)

    return results