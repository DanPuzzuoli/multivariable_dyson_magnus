def hs_ip(x,y):
    """Hilbert-Schmidt inner-product"""
    return (x.conj() * y).sum()

def hs_norm(A):
    return (A.conj() * A).sum().real


def to_hs_norm_func(f):
    """Given a function f of a single argument that returns arrays, return the function
    ||f(x)||^2, where the norm is the Hilbert Schmidt norm.
    """

    def f_norm(x):
        y = f(x)
        return hs_norm(y)

    return f_norm

def get_grape_fidelity_func(U, idx_range=None):
    """Return fidelity function to matrix U.

    idx_range is an optional argument for when the matrix whose fidelity is being measured is
    potentially larger than U - e.g. for when we only care about some subspace
    """

    dim = U.shape[0]

    if idx_range is None:
        idx_range = [0, dim]

    # Function that returns inner product normalized by dim
    fid_func = lambda V: hs_ip(U, V[idx_range[0]:idx_range[1], idx_range[0]:idx_range[1]]) / dim

    return to_hs_norm_func(fid_func)
