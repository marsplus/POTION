import torch

# an iterative method based on Rayleigh 
# quotient to estimate the largest eigenvalue
# and the associated eigenvector
def power_method(mat, Iter=50, output_format='tensor'):
    """
        mat: input matrix, the default format is tensor in PyTorch.
        Iter: #iterations to estimate the leading eigenvector
        output_format: set this to 'array' if the output should be in numpy array
    """

    if mat.sum() == 0:
        return torch.zeros(len(mat))

    n = len(mat)
    x = torch.rand(n) 
    x = x / torch.norm(x, 2)

    flag = 1e-7
    for i in range(Iter):
        x_new = mat @ x
        x_new = x_new / torch.norm(x_new, 2)
        Diff = torch.norm(x_new - x, 2)
        if Diff <= flag:
            break
        x = x_new
    if output_format == 'tensor':
        return x_new
    elif output_format == 'array':
        return x_new.numpy()
    else:
        raise ValueError("Unknown return type.")


def estimate_sym_specNorm(mat, Iter=50, numExp=20):
    """
    mat: input matrix, the default format is tensor in PyTorch.
         Numpy array can also be handled
    Iter: #iterations to estimate the leading eigenvector
    numExp: the #experiments to run in order to estimate the spectral norm
    """
    if mat.sum() == 0:
        return 0

    if type(mat) != torch.Tensor:
        M = torch.tensor(mat, dtype=torch.float32)
    else:
        M = mat.clone()

    ## run the power method multiple times
    ## to make the estimation stable
    spec_norm = 0
    for i in range(numExp):
        v = power_method(M, Iter)
        u = power_method(-M, Iter)
        spec_norm += torch.max( torch.abs(v @ M @ v), torch.abs(u @ (-M) @ u) )
    spec_norm /= numExp
    return spec_norm


def get_submatrix(mat, row_idx, col_idx):
    return torch.index_select(torch.index_select(mat, 0, row_idx), 1, col_idx)


    
