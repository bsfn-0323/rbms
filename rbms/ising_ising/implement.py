import torch
from torch import Tensor
from typing import Tuple
from rbms.custom_fn import log2cosh


def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    effective_field = beta * (hbias + (v @ weight_matrix))
    mh = torch.tanh(effective_field)
    h = 2 * torch.bernoulli(torch.sigmoid(2 * effective_field)) - 1
    return h, mh


def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    effective_field = beta * (vbias + (h @ weight_matrix.T))
    mv = torch.tanh(effective_field)
    v = 2 * torch.bernoulli(torch.sigmoid(2 * effective_field)) - 1
    return v, mv


def _compute_energy(
    v: Tensor,
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
) -> Tensor:
    fields = torch.tensordot(vbias, v, dims=[[0], [1]]) + torch.tensordot(
        hbias, h, dims=[[0], [1]]
    )
    interaction = torch.multiply(
        v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])
    ).sum(1)

    return -fields - interaction


def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = v @ vbias
    exponent = hbias + (v @ weight_matrix)
    log_term = log2cosh(exponent)
    return -field - log_term.sum(1)

def _compute_energy_visibles_gradient(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    local_field = hbias + (v @ weight_matrix)
    tanh_term = torch.tanh(local_field)
    
    grad_vbias = -v
    grad_hbias = -tanh_term
    # grad_weight_matrix = -v.T @ tanh_term
    grad_weight_matrix = None
    return grad_vbias, grad_hbias, grad_weight_matrix

def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = h @ hbias
    exponent = vbias + (h @ weight_matrix.T)
    log_term = log2cosh(exponent)
    return -field - log_term.sum(1)


def _compute_gradient(
    v_data: Tensor,
    mh_data: Tensor,
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    centered: bool = True,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    # Turn the weights of the chains into normalized weights
    chain_weights = w_chain / w_chain.sum()
    w_data_norm = w_data.sum()

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    torch.clamp_(v_data_mean, min=-(1.0 - 1e-7), max=(1.0 - 1e-7))
    h_data_mean = (mh_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
    torch.clamp_(v_gen_mean, min=-(1.0 - 1e-7), max=(1.0 - 1e-7))
    h_gen_mean = (h_chain * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters
    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias

def _compute_var_gradient(
    J1:  Tensor,
    J2:  Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    eta: float,
) -> float:
    
    B = v_chain.size(0)
    
    # 1. Compute energies and local fields
    betaH = _compute_hamiltonian(v_chain, J1, J2)
    local_field = hbias + (v_chain @ weight_matrix)
    tanh_term = torch.tanh(local_field)
    F = _compute_energy_visibles(v_chain, vbias, hbias, weight_matrix)
    deltaE = -betaH + F
    
    # 2. THE CENTERING TRICK
    # Cov(X, Y) = E[X * (Y - E[Y])]. By centering the scalars first, 
    # we bypass calculating the mean of the massive gradient tensors entirely.
    deltaE_c = (deltaE - deltaE.mean()).view(-1, 1)  # Shape: (B, 1)
    # F_c = (F - F.mean()).view(-1, 1)                 # Shape: (B, 1)
    
    # 3. OPTIMIZED WEIGHT GRADIENTS (Pure 2D Matrix Multiplication)
    # v_chain.T is (N_v, B). tanh_term * deltaE_c is (B, N_h).
    # The @ operator resolves to a highly optimized cuBLAS routine.
    grad_weight_matrix = (v_chain.T @ (tanh_term * deltaE_c)) / B
    # entropy_weight_matrix = (v_chain.T @ (tanh_term * F_c)) / B
    
    # 4. OPTIMIZED BIAS GRADIENTS
    # Broadcasting takes care of the element-wise multiplication before the mean
    # grad_hbias = (tanh_term * deltaE_c).mean(dim=0)
    # entropy_hbias = (tanh_term * F_c).mean(dim=0)

    # grad_vbias = (v_chain * deltaE_c).mean(dim=0)
    # entropy_vbias = (v_chain * F_c).mean(dim=0)
    
    # 5. DYNAMIC GAMMA CALCULATION
    # norm_grad = grad_weight_matrix.norm()
    # norm_grad_ent = entropy_weight_matrix.norm()
    target_percentage = eta
    
    # Added 1e-8 epsilon to prevent division by zero in the first step
    loss = 0.5 * (deltaE_c**2).mean()

    # gamma = (norm_grad * target_percentage) / (norm_grad_ent + 1e-8)

    
    # 6. ATTACH GRADIENTS
    weight_matrix.grad = grad_weight_matrix 
    # vbias.grad = grad_vbias + gamma * entropy_vbias
    # hbias.grad = grad_hbias + gamma * entropy_hbias
    vbias.grad = torch.zeros(weight_matrix.shape[0],device = weight_matrix.grad.device)  # Zero out the visible bias gradient to prevent updates
    hbias.grad = torch.zeros(weight_matrix.shape[1],device = weight_matrix.grad.device)  # Zero out the hidden bias gradient to prevent updates
    # The variance loss simplifies neatly with the centered deltaE
    return loss.item()

def _compute_hamiltonian(
    v:Tensor, J1: Tensor, J2:Tensor
) -> Tensor:
    field = v@J1
    interaction = ((v @ J2) * v).sum(1)
    return -field - 0.5*interaction

# def _compute_local_field(
#     v:Tensor, J1: Tensor, J2: Tensor
# ) -> Tensor:
#     return  -v@J2 - J1

def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Tensor | None = None,
):
    num_visibles, _ = weight_matrix.shape
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    # Handle negative number of samples
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        # Dummy mean visible
        mv = torch.ones(size=(num_samples, num_visibles), device=device, dtype=dtype)/2
        v = 2 * torch.bernoulli(mv) - 1
    else:
        # Dummy mean visible
        mv = torch.ones_like(start_v, device=device, dtype=dtype)/2
        v = start_v.to(device=device, dtype=dtype)

    # Initialize chains

    h, mh = _sample_hiddens(v=v, weight_matrix=weight_matrix, hbias=hbias)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4, 
) -> tuple[Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    eps = 1e-4
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    # frequencies = data.mean(0)
    # frequencies = torch.clamp(frequencies, min=-(1.0 - eps), max=(1.0 - eps))
    # vbias = torch.atanh(frequencies).to(device=device, dtype=dtype)
    vbias = torch.zeros(num_visibles,device = device,dtype=dtype)
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return vbias, hbias, weight_matrix
