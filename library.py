"""
Functions no longer in use.
"""

def to_sys_form(G):
    """
    Rewrites the given n x k generator matrix to systematic form.
    TODO this doens't work sometimes when rref gives something where there isn't
    a perfect identiy matrix.
    """
    G = G.T # Transpose so that we reduce cols, not rows.
    G_rref, pivots = Matrix(G).rref()
    G_rref = np.array(G_rref.tolist()).astype(int) % 2
    return G_rref.T # Transpose to go back to G in sys. form.

def get_dual(G):
    """
    Given the n x k generator matrix of a linear code, returns the genetaor
    matrix for its dual code. That is, fist rewrites G in systematic form. Then
    use that to create the (n-k) x n parity check matrix. Then transpose that
    to get the generator of the dual code.
    """
    n, k = G.shape
    G = to_sys_form(G)
    H = np.zeros((n-k, n))
    H[:k, :k] = G[n-k:, :]
    H[:, k:] = np.eye(n-k)
    return H.T # Transpose parity check matrix to get generator for dual code.

def get_weight_k_vector(k, n):
    """
    Returns a uniformly random length n bitstring with weight k.
    """
    v = np.zeros(n)
    v[Random.sample(Random(), range(n), weight)] = 1
    return v
