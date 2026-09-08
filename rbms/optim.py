import numpy as np
import torch
import math
# from ptt.optim.cossim import SGD_cossim
from torch import Tensor
from torch.optim import SGD, Optimizer,Adam

from rbms.classes import EBM
class SGD_cossim(SGD):

    def __init__(
        self,
        params,
        lr=0.001,
        max_lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=True,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        super().__init__(
            params,
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        self.prev_grad = torch.concatenate([p.grad.flatten() for p in params]).flatten()
        self.max_lr = max_lr

    def step(self, closure=None):
        for group in self.param_groups:
            params = group["params"]
            learning_rate = group["lr"]
            curr_grad = torch.concatenate([p.grad.flatten() for p in params]).flatten()
            cosine_similarity = curr_grad @ self.prev_grad
            if cosine_similarity > 1e-6:
                learning_rate *= 1.002
            elif cosine_similarity < -1e-6:
                learning_rate *= 0.998
            group["lr"] = min(self.max_lr, learning_rate)
            self.prev_grad = curr_grad.clone()
        return super().step(closure)
    
class NGD(Optimizer):
    # Added 'update_biases=True' flag to the initialization
    def __init__(self, params, lr=0.001, cg_steps=20, init_reg=1e-6, alpha=1e-3, update_freq=1, warm_start=False, maximize=True, update_biases=True, cossim=False, l2_reg=0.0, n_unif=None, adaptive_reg=False, reg_top_gamma=1, k_top=1, top_warmup=1000):
        defaults = dict(lr=lr, cg_steps=cg_steps, reg=init_reg, alpha=alpha, update_freq=update_freq, warm_start=warm_start, maximize=maximize, update_biases=update_biases, reg_top_gamma=reg_top_gamma, k_top=k_top, top_warmup=top_warmup, step=0)
        super().__init__(params, defaults)
        self.cossim = cossim
        self.lambda_eff = l2_reg
        self.n_unif = n_unif
        self.adaptive_reg = adaptive_reg
        self.reg_top_gamma = reg_top_gamma
        self.k_top = k_top
        self.top_warmup = top_warmup
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['last_dt'] = torch.zeros_like(p.data)
    
    @torch.no_grad()
    def _get_adaptive_reg(self, v_chain_eff, tanh_term, model, base_reg): # <-- Use v_chain_eff
        B = v_chain_eff.size(0)
        
        # Flatten the 3D one-hot tensor to 2D
        v_flat = v_chain_eff.view(B, -1)
        
        v_sq_sum = (v_flat ** 2).sum(dim=1)
        t_sq_sum = (tanh_term ** 2).sum(dim=1)
        
        mean_norm_s_k_sq = (v_sq_sum * t_sq_sum).mean()
        mean_s_w = (v_flat.T @ tanh_term) / B  # Now safe!
        norm_mean_s_sq = (mean_s_w**2).sum()
        
        trace_F = mean_norm_s_k_sq - norm_mean_s_sq
        D = model.weight_matrix.numel() 
        self.trace_F_over_D = trace_F.item() / D  # Store for logging   
        # adaptive_reg = base_reg * self.trace_F_over_D
        return 1e-04 # Ensure strict positivity 
        # return 1e-8
    
    @torch.no_grad()
    def _fvp(self, p_list, v_chain, tanh_term, reg, update_biases, u_top=None, reg_top=0.0):
        B = v_chain.size(0)
        
        # 1. Flatten v_chain to 2D: (B, N_v * N_s)
        v_chain_flat = v_chain.view(B, -1)
        
        if update_biases:
            p_w, p_v, p_h = p_list
            
            # Flatten weights and biases to match
            p_w_flat = p_w.view(-1, p_w.size(-1)) # (N_v * N_s, N_h)
            p_v_flat = p_v.view(-1)               # (N_v * N_s,)
            
            O_dot_p = ((v_chain_flat @ p_w_flat) * tanh_term).sum(dim=1) + (v_chain_flat @ p_v_flat) + (tanh_term @ p_h)
        else:
            p_w = p_list[0]
            p_w_flat = p_w.view(-1, p_w.size(-1))
            O_dot_p = ((v_chain_flat @ p_w_flat) * tanh_term).sum(dim=1)
                  
        O_dot_p_c = O_dot_p - O_dot_p.mean()
        
        # 2. Compute flat gradients
        Sx_w_flat = (v_chain_flat.T @ (tanh_term * O_dot_p_c.unsqueeze(1))) / B
        
        # 3. Reshape back to original parameter shape using .view_as()
        Sx_w = Sx_w_flat.view_as(p_w)
        # --- anisotropic damping on the top visible mode(s) ---
        if u_top is not None and reg_top != 0.0:
            p_w_flat = p_w.view(-1, p_w.size(-1))        # (D_v, N_h)
            proj = u_top @ (u_top.T @ p_w_flat)          # (D_v, N_h)
            Sx_w = Sx_w + reg_top * proj.view_as(p_w)
        # ------------------------------------------------------

        if update_biases:
            Sx_v_flat = (v_chain_flat.T @ O_dot_p_c) / B
            Sx_v = Sx_v_flat.view_as(p_v)
            Sx_h = (tanh_term.T @ O_dot_p_c) / B
            return [Sx_w + reg * p_w, Sx_v + reg * p_v, Sx_h + reg * p_h]
        else:
            return [Sx_w + reg * p_w]
    @torch.no_grad()
    def step(self, v_chain, model, scale=1, closure=None): 
        for group in self.param_groups:
            group["step"] += 1
            params = group["params"]
            # lr = group["lr"]
            update_biases = group["update_biases"]
            max_lr = 5/model.weight_matrix.numel()**0.5
            g = [p.grad.clone() for p in params]
            
            if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                grad_v, grad_h, _ = model.compute_energy_visible_gradient(v_chain)
                
                v_chain_eff = -grad_v
                tanh_term = -grad_h
                if self.adaptive_reg:
                    adaptive_term = scale * self._get_adaptive_reg(v_chain, tanh_term, model, group["alpha"])
                    self.reg = max(adaptive_term, group["reg"])
                else:
                    self.reg = group["reg"]

                # FLAG LOGIC: Filter parameters fed to the Conjugate Gradient solver
                # top-k left singular vectors of W (visible-side modes)
                W_flat = model.weight_matrix.detach().view(-1, model.weight_matrix.size(-1))
                if self.reg_top_gamma > 0.0 and group["step"] > self.top_warmup:
                    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
                    u_top = U[:, :self.k_top].contiguous()          # (D_v, k)
                    self.reg_top = self.reg_top_gamma * W_flat.size(0) * self.trace_F_over_D
                    # self.reg_top = self.reg_top_gamma * W_flat.size(0) 
                    # self.reg_top = self.reg_top_gamma
                    # ipr = 1/torch.sum(Vh[0]**4)  # inverse participation ratio
                    # ratio = model.weight_matrix.size(0)*ipr
                    # # nu from the CONDENSED ferromagnetic mode only.
                    # m2 = (v_chain.mean(-1)**2).mean()          # <s>^2 proxy, see caveats
                    # nu = ratio * S[0]**2 * m2
                    # denom = 1 - 2/3 * nu
                    # delta = 0.25
                    # if denom < delta:
                    #     self.reg_top = self.reg_top_gamma * (delta - denom)   # inject only the shortfall
                    # else:
                    #     self.reg_top = 0.0
                    self.sv_top = S[:self.k_top].tolist()           # log this
                else:
                    u_top = None
                    self.reg_top = 0.0

                if update_biases:
                    active_params = params
                    active_grads = g
                else:
                    active_params = [p for p in params if p is model.weight_matrix]
                    active_grads = [p.grad.clone() for p in active_params]
                
                if group["warm_start"] and group["step"] > 1:
                    delta_theta = [self.state[p]['last_dt'].clone() for p in active_params]
                    S_dt = self._fvp(delta_theta, v_chain_eff, tanh_term, self.reg, update_biases, u_top=u_top, reg_top=self.reg_top)
                    residual = [g_i - s_i for g_i, s_i in zip(active_grads, S_dt)]
                else:
                    delta_theta = [torch.zeros_like(p) for p in active_params]
                    residual = [g_i.clone() for g_i in active_grads]
                    
                p_vec = [r_i.clone() for r_i in residual]
                r_dot_r = sum(torch.sum(r * r) for r in residual)
                
                # Fix: Do not add an artificial 1e-8. If it's exactly 0, replace it with machine epsilon.
                initial_r_norm = torch.sqrt(r_dot_r).item()
                if initial_r_norm == 0:
                    initial_r_norm = 1e-20
                    
                current_r_norm = initial_r_norm
                self.cg_step = 0
                
                # while (current_r_norm / initial_r_norm) > 0.05:
                #     if self.cg_step >= group["cg_steps"]:
                #         # print("CG broke due to reaching max iterations.")
                #         break
                for _ in range(group["cg_steps"]):
                    S_p = self._fvp(p_vec, v_chain_eff, tanh_term, self.reg, update_biases, u_top=u_top, reg_top=self.reg_top)
                    p_Sp = sum(torch.sum(pv * spv) for pv, spv in zip(p_vec, S_p))
                    
                    if p_Sp.item() <= 0:
                        print("CG broke due to non-positive curvature.")
                        break
                        
                    alpha = (r_dot_r / p_Sp).item()
                    
                    for dt, pv in zip(delta_theta, p_vec):
                        dt.add_(pv, alpha=alpha)
                        
                    for r, spv in zip(residual, S_p):
                        r.sub_(spv, alpha=alpha)
                    
                    new_r_dot_r = sum(torch.sum(r * r) for r in residual)
                    current_r_norm = torch.sqrt(new_r_dot_r).item()
                    
                    if new_r_dot_r.item() < 0:
                        print("CG broke due to tiny residual norm.")
                        break
                        
                    beta = (new_r_dot_r / r_dot_r).item()
                    
                    for pv, r in zip(p_vec, residual):
                        pv.mul_(beta).add_(r)
                        
                    r_dot_r = new_r_dot_r
                    self.cg_step += 1
                
                for p_tensor, dt in zip(active_params, delta_theta):
                    self.state[p_tensor]['last_dt'].copy_(dt)
            else:
                delta_theta = g
                active_params = params
                active_grads = g
            
            if self.cossim:
                # # --- Added Cosine Similarity ---
                dot_product = sum(torch.sum(dt * g_i) for dt, g_i in zip(delta_theta, active_grads))
                norm_dt = torch.sqrt(sum(torch.sum(dt ** 2) for dt in delta_theta))
                norm_g = torch.sqrt(sum(torch.sum(g_i ** 2) for g_i in active_grads))

                self.cos_sim = (dot_product / (norm_dt * norm_g + 1e-20)).item()
                # -------------------------------
                # --- Raise learning rate based on cosine similarity ---
                # Increases lr if the similarity is positive (up to 2x if perfectly aligned)

                group['lr'] *=  1 + 0.005 * self.cos_sim  # Scale increase by cosine similarity


                group['lr'] = min(group['lr'], max_lr)
            # ------------------------------------------------------

            # --- ℓ2 regularization of effective-coupling space ---
            # Penalty R = (λ/2) Var_u[E], gradient ∇_θ R = λ Cov_u(∂_θ E, E).
            # We subtract ∇R from delta_theta (un-preconditioned, after CG) so that
            # the net parameter update descends on R and Var_u[E] decreases.
            # Uniform samples come from the model's pre-allocated buffer (Ising ±1).
            if self.lambda_eff != 0.0:
                n_u = self.n_unif if self.n_unif is not None else v_chain.size(0)
                ridx = torch.randint(0, model.random_chain_buffer.size(0), (n_u,), device=model.device)
                v_unif = model.random_chain_buffer[ridx]
                E_unif = model.compute_energy_visibles(v_unif)
                E_c = E_unif - E_unif.mean()                            # (n_u,)
                tanh_u = torch.tanh(model.hbias + v_unif @ model.weight_matrix)  # (n_u, N_h)
                for i, p_tensor in enumerate(active_params):
                    if p_tensor is model.weight_matrix:
                        # Cov_u(∂_W E, E) = -(v_unif.T @ (tanh_u * E_c[:,None])) / n_u
                        g_reg = self.lambda_eff * ((v_unif.T @ (tanh_u * E_c.unsqueeze(1))) / n_u)
                        delta_theta[i] = delta_theta[i] + g_reg
                    elif update_biases and p_tensor is model.vbias:
                        # Cov_u(∂_vbias E, E) = -(v_unif.T @ E_c) / n_u
                        g_reg = self.lambda_eff * ((v_unif.T @ E_c) / n_u)
                        delta_theta[i] = delta_theta[i] + g_reg
                    elif update_biases and p_tensor is model.hbias:
                        # Cov_u(∂_hbias E, E) = -(tanh_u.T @ E_c) / n_u
                        g_reg = self.lambda_eff * ((tanh_u.T @ E_c) / n_u)
                        delta_theta[i] = delta_theta[i] + g_reg
            # -----------------------------------------------------

            # --- per-step KL diagnostic: eps = 1/2 lr^2 * dt^T F dt ---
            g_dot_x = sum(torch.sum(g_i * dt) for g_i, dt in zip(active_grads, delta_theta))
            x_dot_x = sum(torch.sum(dt * dt) for dt in delta_theta)
            xFx = (g_dot_x - self.reg * x_dot_x).clamp_min(0.0)
            reg_quad = self.reg * x_dot_x
            if getattr(self, "reg_top", 0.0) != 0.0 and u_top is not None:
                dw = delta_theta[0].view(-1, delta_theta[0].size(-1))
                reg_quad = reg_quad + self.reg_top * (u_top.T @ dw).pow(2).sum()
            xFx = (g_dot_x - reg_quad).clamp_min(0.0)
            self.epsilon = (0.5 * group["lr"]**2 * xFx)
            
            direction = 1 if group["maximize"] else -1
            for p_tensor, dt in zip(active_params, delta_theta):
                p_tensor.add_(dt, alpha=direction * group['lr'])

def setup_optim(optim: str, args: dict, params: EBM) -> list[Optimizer]:
    match args["optim"]:
        case "sgd":
            optim_class = SGD
        case "cossim":
            optim_class = SGD_cossim
        case "adam":
            optim_class = Adam
        case "ngd":
            optim_class = NGD
        case _:
            print(f"Unrecognized optimizer {args['optim']}, falling back to SGD.")
            optim_class = SGD
    learning_rate = args["learning_rate"]
    max_lr = args["max_lr"]
    if args["scale_lr"]:
        learning_rate /= np.sqrt(params.effective_number_variables)
        max_lr /= np.sqrt(params.effective_number_variables)

    if args["optim"] in ["ngd"]:
        if not isinstance(learning_rate, Tensor):
            learning_rate = torch.tensor([learning_rate])
            
        optimizer = [
            optim_class(
                params.parameters(), 
                lr=learning_rate[0],
                update_freq=1, 
                warm_start=True,
                maximize=True,
                update_biases=True,
                cossim = args["ngd_cossim"],
                adaptive_reg = True,
                l2_reg = args["L2_effective"],
                top_warmup=50,
                reg_top_gamma=0.5
            )
        ]
    else:
        if args["mult_optim"]:
            if not isinstance(learning_rate, Tensor):
                learning_rate = torch.tensor([learning_rate] * len(params.parameters()))
            optimizer = [
                optim_class(
                    [p],
                    lr=learning_rate[i],
                    max_lr=args['max_lr'],
                    maximize=True,
                )
                for i, p in enumerate(params.parameters())
            ]
        else:
            if not isinstance(learning_rate, Tensor):
                learning_rate = torch.tensor([learning_rate])
            optimizer = [
                optim_class(
                    params.parameters(),
                    lr=learning_rate[0],
                    maximize=True,
                )
            ]
    for opt in optimizer:
        if isinstance(opt, SGD_cossim):
            opt.max_lr = max_lr

    if args["optim"] == "nag":
        optimizer = [
            SGD(
                opt.param_groups[0]["params"],
                lr=opt.param_groups[0]["lr"],
                maximize=True,
                momentum=0.9,
                nesterov=True,
            )
            for opt in optimizer
        ]

    return optimizer
