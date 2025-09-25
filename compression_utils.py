import torch
from torch.types import Tensor


@torch.no_grad
def sqrt_M(M: Tensor, ridge_lambda=1e-4) -> Tensor:
    """
    Warning: Be weary of input being of precision torch.float64, may be because ridge lambda was too low,
    but this broke the result in some cases.
    The following is the code I originally used for sqrt_M. Its MUCH slower.
    However, note the following marginal ppl differerence against opt-1.3b WikiText2:
    Perplexity with eigenvalues:  16.44
    Perplexity with below method: 16.35

        # _C = cov[layer].to(device="cpu")
        # C = _C + torch.eye(_C.shape[0], device="cpu") * ridge_lambda

        # # === This is the most computationally exspensive task ======
        # # Can possible make cov_x a numpy array from the beginning to reduce this
        # C_np = C.detach().cpu().numpy()  # should already be on cpu but just in case
        # sqrt_C_np = sqrtm(C_np).real
        # sqrt_C = torch.from_numpy(sqrt_C_np).to(dtype=torch.float32, device="cuda")
    """

    M_reg = M + torch.eye(M.shape[0], device=M.device) * ridge_lambda
    eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
    sqrt_M: Tensor = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    return sqrt_M.to(dtype=M.dtype)
