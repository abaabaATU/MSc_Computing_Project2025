import numpy as np

def TE_calculator(A, V, p, lags=1):
    """
    Calculate Transfer Entropy (TE) based on VAR model
    
    Parameters:
        A: VAR coefficient matrix (p*n*n)
        V: Residual covariance matrix (n*n)
        p: VAR model order
        lags: Time lag for TE calculation (default=1)
    
    Returns:
        te_matrix: Transfer entropy matrix (n*n)
    """
    n = A.shape[1]  # Number of brain regions
    te_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate TE for X→Y
                sigma_cond = V[j,j] - V[j,i]**2 / V[i,i]
                sigma_full = V[j,j]
                te_matrix[i,j] = 0.5 * np.log(sigma_cond / sigma_full)
    
    return te_matrix