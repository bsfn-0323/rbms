import torch
from torch import Tensor
from torch.nn.functional import softmax
from typing import Tuple
from rbms.custom_fn import one_hot


def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )
    mh = torch.sigmoid(beta * (hbias + v_oh @ weight_matrix_oh))
    h = torch.bernoulli(mh).to(weight_matrix.dtype)
    return h, mh


def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    num_visibles, num_states, _ = weight_matrix.shape
    mv = torch.softmax(
        beta * (vbias + torch.tensordot(h, weight_matrix, dims=[[1], [2]])),
        dim=-1,
    )
    v = (
        torch.multinomial(mv.view(-1, num_states), 1)
        .view(-1, num_visibles)
        .to(weight_matrix.dtype)
    )
    return v, mv


def _compute_energy(
    v: Tensor, h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )
    vbias_oh = vbias.flatten()
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    fields = (v_oh @ vbias_oh) + (h @ hbias)
    interaction = ((v_oh @ weight_matrix_oh) * h).sum(1)
    return -fields - interaction


def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )

    vbias_oh = vbias.flatten()
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    field = v_oh @ vbias_oh
    exponent = hbias + (v_oh @ weight_matrix_oh)
    log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)
    return -field - log_term.sum(1)


def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    field = h @ hbias
    arg_lse = vbias + torch.tensordot(h, weight_matrix, dims=[[1], [2]])
    lse = torch.logsumexp(arg_lse, dim=2).sum(1)
    return -field - lse


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
):
    w_data = w_data.view(-1, 1, 1)
    w_chain = w_chain.view(-1, 1, 1)
    int_dtype = torch.int32
    dtype = weight_matrix.dtype
    num_states = weight_matrix.shape[1]

    # One-hot representation of the data
    v_data_one_hot = one_hot(v_data.to(int_dtype), num_classes=num_states, dtype=dtype)
    v_gen_one_hot = one_hot(v_chain.to(int_dtype), num_classes=num_states, dtype=dtype)

    # Turn the weights of the chains into normalized weights
    chain_weights = softmax(-w_chain, dim=0)
    w_chain_norm = chain_weights.sum()
    w_data_norm = w_data.sum()
    # Averages over data and generated samples
    v_data_mean = (v_data_one_hot * w_data).sum(0) / w_data_norm
    h_data_mean = (mh_data * w_data.view(-1, 1)).sum(0) / w_data_norm
    v_gen_mean = (v_gen_one_hot * chain_weights).sum(0) / w_chain_norm
    h_gen_mean = (h_chain * chain_weights.view(-1, 1)).sum(0) / w_chain_norm
    torch.clamp_(v_data_mean, min=1e-7, max=(1.0 - 1e-7))
    torch.clamp_(v_gen_mean, min=1e-7, max=(1.0 - 1e-7))
    if centered:
        # Centered variables
        v_data_centered = v_data_one_hot - v_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_gen_one_hot - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            torch.tensordot(
                v_data_centered,
                h_data_centered,
                dims=[[0], [0]],
            )
            / v_data.shape[0]
            - torch.tensordot(
                v_gen_centered,
                h_gen_centered,
                dims=[[0], [0]],
            )
            / v_chain.shape[0]
        )
        grad_vbias = (
            v_data_mean
            - v_gen_mean
            - torch.tensordot(grad_weight_matrix, h_data_mean, dims=[[2], [0]])
        )
        grad_hbias = (
            h_data_mean
            - h_gen_mean
            - torch.tensordot(v_data_mean, grad_weight_matrix, dims=[[0, 1], [0, 1]])
        )
    else:
        # Gradient
        grad_weight_matrix = (
            torch.tensordot(
                v_data_one_hot,
                mh_data,
                dims=[[0], [0]],
            )
            / v_data.shape[0]
            - torch.tensordot(
                v_gen_one_hot,
                h_chain,
                dims=[[0], [0]],
            )
            / v_chain.shape[0]
        )

        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias

def _compute_hamiltonian(v_oh: Tensor, J1: Tensor, J2: Tensor) -> Tensor:
    """
    v_oh: (B, N_v, N_s)
    J1: (N_v, N_s)
    J2: (N_v, N_v)
    """
    # field = (v_oh * J1).sum(dim=(1, 2))
    
    # v_oh @ v_oh.T gives a (B, N_v, N_v) matrix where entry (i,j) is 1 if they share a color
    color_matches = torch.bmm(v_oh, v_oh.transpose(1, 2))
    interaction = (color_matches * J2).sum(dim=(1, 2))
    
    return - 0.5 * interaction

def _compute_var_gradient(
    J1: Tensor,
    J2: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    eta: float,
) -> float:
    B = v_chain.size(0)
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    
    # 3D one-hot for the Potts Hamiltonian: (B, N_v, N_s)
    v_oh_3d = one_hot(v_chain.to(torch.int32), num_classes=num_states, dtype=dtype)
    
    # Flat one-hot for the RBM linear algebra: (B, N_v * N_s)
    v_oh = v_oh_3d.view(-1, num_visibles * num_states)
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    
    # 1. Compute energies and local fields
    betaH = _compute_hamiltonian(v_oh_3d, J1, J2)
    local_field = hbias + (v_oh @ weight_matrix_oh)

    sigmoid_term = torch.sigmoid(local_field)
    
    # F uses the existing function (which internally one-hots v_chain)
    F = _compute_energy_visibles(v_chain, vbias, hbias, weight_matrix)
    deltaE = -betaH + F
    
    # 2. THE CENTERING TRICK
    deltaE_c = (deltaE - deltaE.mean()).view(-1, 1)  # Shape: (B, 1)
    # F_c = (F - F.mean()).view(-1, 1)                 # Shape: (B, 1)
    
    # 3. OPTIMIZED WEIGHT GRADIENTS
    # v_oh.T is (N_v * N_s, B). sigmoid_term * deltaE_c is (B, N_h).
    grad_weight_matrix_oh = (v_oh.T @ (sigmoid_term * deltaE_c)) / B
    # entropy_weight_matrix_oh = (v_oh.T @ (sigmoid_term * F_c)) / B
    
    # Reshape back to (N_v, N_s, N_h)
    grad_weight_matrix = grad_weight_matrix_oh.view(num_visibles, num_states, num_hiddens)
    # entropy_weight_matrix = entropy_weight_matrix_oh.view(num_visibles, num_states, num_hiddens)
    
    # 4. OPTIMIZED BIAS GRADIENTS
    grad_hbias = (sigmoid_term * deltaE_c).mean(dim=0)
    # entropy_hbias = (sigmoid_term * F_c).mean(dim=0)

    grad_vbias_oh = (v_oh * deltaE_c).mean(dim=0)
    # entropy_vbias_oh = (v_oh * F_c).mean(dim=0)
    
    # Reshape vbias gradients to (N_v, N_s)
    grad_vbias = grad_vbias_oh.view(num_visibles, num_states)
    # entropy_vbias = entropy_vbias_oh.view(num_visibles, num_states)
    
    # 5. DYNAMIC GAMMA CALCULATION
    # norm_grad = grad_weight_matrix.norm()
    # norm_grad_ent = entropy_weight_matrix.norm()
    # target_percentage = eta
    
    loss = 0.5 * (deltaE_c**2).mean()
    # gamma = (norm_grad * target_percentage) / (norm_grad_ent + 1e-8)
    
    # 6. ATTACH GRADIENTS
    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias 
    
    return loss.item()

def _compute_energy_visibles_gradient(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    
    # 3D one-hot for the visible biases: (B, N_v, N_s)
    v_oh_3d = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype)
    
    # Flattened one-hot for matrix multiplication: (B, N_v * N_s)
    v_oh = v_oh_3d.view(-1, num_visibles * num_states)
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    
    local_field = hbias + (v_oh @ weight_matrix_oh)
    sigmoid_term = torch.sigmoid(local_field)
    
    # Gradients (preserving the batch dimension)
    grad_vbias = -v_oh_3d
    grad_hbias = -sigmoid_term
    
    # To match your ising_ising structure, leaving weight matrix grad as None
    # Full computation would be: -torch.bmm(v_oh_3d.view(B, -1, 1), sigmoid_term.unsqueeze(1))
    grad_weight_matrix = None
    
    return grad_vbias, grad_hbias, grad_weight_matrix

def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Tensor | None = None,
):
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    if start_v is None:
        v = torch.randint(
            0,
            num_states,
            size=(num_samples, num_visibles),
            device=weight_matrix.device,
            dtype=weight_matrix.dtype,
        )
    else:
        v = start_v.to(weight_matrix.dtype)
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(
        v.to(torch.int32), num_classes=num_states, dtype=weight_matrix_oh.dtype
    ).view(-1, num_visibles * num_states)
    mv = torch.zeros(v.shape[0], v.shape[1], num_states)
    mh = torch.sigmoid(hbias + v_oh @ weight_matrix_oh)
    h = torch.bernoulli(mh)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
    num_states: int = None,
) -> tuple[Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    eps = 1e-7
    if num_states is None:
        num_states = int(torch.max(data) + 1)

    # num_states = data.get_num_states()
    all_states = torch.arange(num_states).reshape(-1, 1, 1).to(data.device)
    frequencies = (data == all_states).type(torch.float32).mean(1).to(device)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (
        (torch.log(frequencies) - 1.0 / num_states * torch.sum(torch.log(frequencies), 0))
        .to(device=device, dtype=dtype)
        .T
    )
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    weight_matrix = (
        torch.randn(
            size=(num_visibles, num_states, num_hiddens), device=device, dtype=dtype
        )
        * var_init
    )
    return vbias, hbias, weight_matrix


def _zero_sum_gauge(vbias: Tensor, hbias: Tensor, weight_matrix: Tensor):
    mean_W = weight_matrix.mean(1, keepdim=True)
    weight_matrix -= mean_W
    hbias += mean_W.squeeze().sum(0)
    vbias -= vbias.mean(1, keepdim=True)
    return vbias, hbias, weight_matrix
