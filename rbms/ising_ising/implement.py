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
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor, average_over_batch: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    # if average_over_batch:
    #     local_field = hbias + (v @ weight_matrix)
    #     tanh_term = torch.tanh(local_field)
    # else:
    #     local_field = hbias[None, :] + (v @ weight_matrix)
    #     tanh_term = torch.tanh(local_field)

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
    l2_reg: float,
    random_chain: Tensor | None = None,
) -> float:
    
    B = v_chain.size(0)

    if random_chain is None:
        random_chain = torch.bernoulli(torch.full_like(v_chain, 0.5)) * 2 - 1
        random_chain = random_chain.to(device=v_chain.device, dtype=v_chain.dtype)

    # 1. Compute target model energies
    betaH = _compute_hamiltonian(v_chain, J1, J2)

    grad_F_v,grad_F_h,_ = _compute_energy_visibles_gradient(
        v=v_chain,
        vbias=vbias,
        hbias=hbias,
        weight_matrix=weight_matrix,
    )
    F = _compute_energy_visibles(v_chain, vbias, hbias, weight_matrix)

    # local_field = hbias + (v_chain @ weight_matrix)
    # tanh_term = torch.tanh(local_field)
    
    # F_uniform = _compute_energy_visibles(random_chain, vbias, hbias, weight_matrix)
    deltaE = -betaH + F
    
    # 2. THE CENTERING TRICK
    deltaE_c = (deltaE - deltaE.mean()).view(-1, 1)  # Shape: (B, 1)
    
    # 3. OPTIMIZED WEIGHT GRADIENTS (Pure 2D Matrix Multiplication)
    grad_weight_matrix = (grad_F_v.T @ (grad_F_h * deltaE_c)) / B
    
    # 4. OPTIMIZED BIAS GRADIENTS
    # Broadcasting takes care of the element-wise multiplication before the mean
    grad_hbias = -(grad_F_h * deltaE_c).mean(dim=0)
    grad_vbias = -(grad_F_v * deltaE_c).mean(dim=0)

    # grad_F_v_u,grad_F_h_u,_ = _compute_energy_visibles_gradient(
    #     v=random_chain,
    #     vbias=vbias,
    #     hbias=hbias,
    #     weight_matrix=weight_matrix,
    # )
    # F_centered = F_uniform - F_uniform.mean() # (B,)
    # Since F_centered sums to zero, centering grad_F_weight_matrix across the
    # batch contributes nothing to the contraction below, so we skip building
    # the (B, N_v, N_h) outer-product tensor and contract via two matmuls.
    # l2_weight_matrix = (grad_F_v * F_centered[:, None]).T @ grad_F_h / B
    # l2_vbias = -grad_F_v.T @ F_centered / B
    # l2_hbias = -grad_F_h.T @ F_centered / B 


    # Added 1e-8 epsilon to prevent division by zero in the first step
    loss = 0.5 * (deltaE_c**2).mean()

    # gamma = (norm_grad * target_percentage) / (norm_grad_ent + 1e-8)

    
    # 6. ATTACH GRADIENTS
    # Only the KL gradient goes into .grad so that a natural-gradient optimizer
    # (which applies F^{-1} built from p_rbm statistics) does not inadvertently
    # precondition the variance term, whose geometry lives under the uniform measure.
    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias
    # vbias.grad = torch.zeros_like(vbias)
    # hbias.grad = torch.zeros_like(hbias)

    # Variance regularization gradients returned separately; the caller must add
    # them to the parameter update *after* any natural-gradient step.
    # var_grads = (
    #     l2_reg * l2_weight_matrix,   # ascent direction for W
    #     l2_reg * l2_vbias,            # ascent direction for vbias
    #     l2_reg * l2_hbias,            # ascent direction for hbias
    # )
    return loss.item(), deltaE

# def _compute_hamiltonian(
#     v:Tensor, J1: Tensor, J2:Tensor
# ) -> Tensor:
#     field = v@J1
#     interaction = ((v @ J2) * v).sum(1)
#     return -field - 0.5*interaction

def _compute_hamiltonian(
    v:Tensor, J1: Tensor, J2:Tensor, J3:Tensor 
) -> Tensor:
    field = v@J1
    # interaction = ((v @ J2) * v).sum(1)
    interaction = (J2.max() * v * torch.roll(v, -1, dims=-1)).sum(-1)
    # if J3 is not None:
        # interaction_3 = torch.einsum("bi,bj,bk,ijk->b", v,v,v,J3)
    interaction_3= (J2.max()* v * torch.roll(v, -1, dims=-1) * torch.roll(v, -2, dims=-1)).sum(-1)
    # else:
    #     interaction_3 = 0.0
    return -field - interaction - interaction_3

# def _compute_hamiltonian(
#     v:Tensor, J1: Tensor, J2:Tensor
# ) -> Tensor:
#     field = v@J1
#     # interaction = ((v @ J2) * v).sum(1)
#     interaction = (J2.max() * v * torch.roll(v, -1, dims=-1)).sum(-1)
#     # if J3 is not None:
#         # interaction_3 = torch.einsum("bi,bj,bk,ijk->b", v,v,v,J3)
#     interaction_3= (J2.max()* v * torch.roll(v, -1, dims=-1) * torch.roll(v, -2, dims=-1)).sum(-1)
#     # else:
#     #     interaction_3 = 0.0
#     return -field - interaction - 0.5*interaction_3

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
