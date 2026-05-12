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

# class NGD(Optimizer):
#     def __init__(self, params, lr=0.001, cg_steps=50, init_reg=1, update_freq=1, warm_start=False, maximize=True):
#         defaults = dict(lr=lr, cg_steps=cg_steps, reg=init_reg, update_freq=update_freq, warm_start=warm_start, maximize=maximize, step=0)
#         super().__init__(params, defaults)
        
#         for group in self.param_groups:
#             for p in group['params']:
#                 self.state[p]['last_dt'] = torch.zeros_like(p.data)
    
#     @torch.no_grad()
#     def _get_adaptive_reg(self, v_chain, tanh_term, model, base_reg):
#         B = v_chain.size(0)
#         v_sq_sum = (v_chain ** 2).sum(dim=1)
#         t_sq_sum = (tanh_term ** 2).sum(dim=1)
        
#         # Only W terms for the trace
#         mean_norm_s_k_sq = (v_sq_sum * t_sq_sum).mean()
#         mean_s_w = (v_chain.T @ tanh_term) / B
#         norm_mean_s_sq = (mean_s_w**2).sum()
        
#         trace_F = mean_norm_s_k_sq - norm_mean_s_sq
#         D = model.weight_matrix.numel() # Only dimension of W
        
#         adaptive_reg = base_reg * (trace_F ).item()/D

#         # adaptive_reg = base_reg 
#         return adaptive_reg
    
#     @torch.no_grad()
#     def _fvp(self, p_list, v_chain, tanh_term, reg):
#         # Unpack parameters based on model.parameters() ordering: 
#         # (weight_matrix, vbias, hbias)
#         p_w, p_v, p_h = p_list
            
#         O_dot_p = ((v_chain @ p_w) * tanh_term).sum(dim=1) + \
#                   (v_chain @ p_v) + \
#                   (tanh_term @ p_h)
                  
#         O_dot_p_c = O_dot_p - O_dot_p.mean()
        
#         B = v_chain.size(0)
        
#         Sx_w = (v_chain.T @ (tanh_term * O_dot_p_c.unsqueeze(1))) / B
#         Sx_v = (v_chain.T @ O_dot_p_c) / B
#         Sx_h = (tanh_term.T @ O_dot_p_c) / B
                
#         return [Sx_w + reg * p_w, Sx_v + reg * p_v, Sx_h + reg * p_h]

#     @torch.no_grad()
#     def step(self, v_chain, model, scale=1, closure=None): # <-- Added 'model' back
#         for group in self.param_groups:
#             group["step"] += 1
#             params = group["params"]
#             lr = group["lr"]
#             # reg = scale * group["reg"]
#             g = [p.grad.clone() for p in params]
            
#             if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                
#                 # Use 'model' directly from the argument
#                 grad_v, grad_h, _ = model.compute_energy_visible_gradient(v_chain)
                
#                 v_chain_eff = -grad_v
#                 tanh_term = -grad_h
#                 self.reg = scale*self._get_adaptive_reg(v_chain, tanh_term, model, group["reg"])
                
#                 if group["warm_start"] and group["step"] > 1:
#                     # Warm Start: Initialize with previous Natural Gradient
#                     delta_theta = [self.state[p]['last_dt'].clone() for p in params]
#                     S_dt = self._fvp(delta_theta, v_chain_eff, tanh_term, self.reg)
#                     residual = [g_i - s_i for g_i, s_i in zip(g, S_dt)]
#                 else:
#                     # Cold Start
#                     delta_theta = [torch.zeros_like(p) for p in params]
#                     residual = [g_i.clone() for g_i in g]
                    
#                 p_vec = [r_i.clone() for r_i in residual]
#                 r_dot_r = sum(torch.sum(r * r) for r in residual)
#                 initial_r_norm = torch.sqrt(r_dot_r).item() + 1e-8
#                 current_r_norm = initial_r_norm
#                 self.cg_step = 0
                
#                 # Keep looping while the relative error is larger than the tolerance
#                 while (current_r_norm / initial_r_norm) > 0.001:
#                 # for _ in range(group["cg_steps"]):
#                     if self.cg_step >= group["cg_steps"]:
#                         break
                        
#                     S_p = self._fvp(p_vec, v_chain_eff, tanh_term, self.reg)
#                     p_Sp = sum(torch.sum(pv * spv) for pv, spv in zip(p_vec, S_p))
                    
#                     if p_Sp.item() <= 1e-20:
#                         print(f"Broke: {p_Sp.item()}")
#                         break
                        
#                     alpha = (r_dot_r / p_Sp).item()
                    
#                     for dt, pv in zip(delta_theta, p_vec):
#                         dt.add_(pv, alpha=alpha)
                        
#                     for r, spv in zip(residual, S_p):
#                         r.sub_(spv, alpha=alpha)
                    
#                     new_r_dot_r = sum(torch.sum(r * r) for r in residual)
#                     current_r_norm = torch.sqrt(new_r_dot_r).item()
                    
#                     if new_r_dot_r.item() < 1e-20:
#                         print(f"Broke: {new_r_dot_r.item()}")
#                         break
                        
#                     beta = (new_r_dot_r / r_dot_r).item()
                    
#                     for pv, r in zip(p_vec, residual):
#                         pv.mul_(beta).add_(r)
                        
#                     r_dot_r = new_r_dot_r
#                     self.cg_step += 1
                
#                 # Save the Natural Gradient for the next warm start
#                 for p_tensor, dt in zip(params, delta_theta):
#                     if p_tensor is model.weight_matrix: # <-- Check against 'model.weight_matrix'
#                         self.state[p_tensor]['last_dt'].copy_(dt)
#             else:
#                 delta_theta = g
                
#             direction = 1 if group["maximize"] else -1
#             for p_tensor, dt in zip(params, delta_theta):
#                 if p_tensor is model.weight_matrix: # <-- Check against 'model.weight_matrix'
#                     p_tensor.add_(dt, alpha=direction * lr)

class NGD(Optimizer):
    # Added 'update_biases=True' flag to the initialization
    def __init__(self, params, lr=0.001, cg_steps=10, init_reg=1, update_freq=1, warm_start=False, maximize=True, update_biases=True):
        defaults = dict(lr=lr, cg_steps=cg_steps, reg=init_reg, update_freq=update_freq, warm_start=warm_start, maximize=maximize, update_biases=update_biases, step=0)
        super().__init__(params, defaults)
        
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
        
        adaptive_reg = base_reg * trace_F.item() / D
        return adaptive_reg # Ensure strict positivity 
        # return 1e-6
    
    @torch.no_grad()
    def _fvp(self, p_list, v_chain, tanh_term, reg, update_biases):
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
            
            g = [p.grad.clone() for p in params]
            
            if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                grad_v, grad_h, _ = model.compute_energy_visible_gradient(v_chain)
                
                v_chain_eff = -grad_v
                tanh_term = -grad_h
                self.reg = scale * self._get_adaptive_reg(v_chain, tanh_term, model, group["reg"])
                
                # FLAG LOGIC: Filter parameters fed to the Conjugate Gradient solver
                if update_biases:
                    active_params = params
                    active_grads = g
                else:
                    active_params = [p for p in params if p is model.weight_matrix]
                    active_grads = [p.grad.clone() for p in active_params]
                
                if group["warm_start"] and group["step"] > 1:
                    delta_theta = [self.state[p]['last_dt'].clone() for p in active_params]
                    S_dt = self._fvp(delta_theta, v_chain_eff, tanh_term, self.reg, update_biases)
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
                
                while (current_r_norm / initial_r_norm) > 0.01:
                    if self.cg_step >= group["cg_steps"]:
                        # print("CG broke due to reaching max iterations.")
                        break
                # for _ in range(group["cg_steps"]):
                    S_p = self._fvp(p_vec, v_chain_eff, tanh_term, self.reg, update_biases)
                    p_Sp = sum(torch.sum(pv * spv) for pv, spv in zip(p_vec, S_p))
                    
                    if p_Sp.item() <= 1e-20:
                        print("CG broke due to non-positive curvature.")
                        break
                        
                    alpha = (r_dot_r / p_Sp).item()
                    
                    for dt, pv in zip(delta_theta, p_vec):
                        dt.add_(pv, alpha=alpha)
                        
                    for r, spv in zip(residual, S_p):
                        r.sub_(spv, alpha=alpha)
                    
                    new_r_dot_r = sum(torch.sum(r * r) for r in residual)
                    current_r_norm = torch.sqrt(new_r_dot_r).item()
                    
                    if new_r_dot_r.item() < 1e-20:
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
            
            # --- Added Cosine Similarity ---
            dot_product = sum(torch.sum(dt * g_i) for dt, g_i in zip(delta_theta, active_grads))
            norm_dt = torch.sqrt(sum(torch.sum(dt ** 2) for dt in delta_theta))
            norm_g = torch.sqrt(sum(torch.sum(g_i ** 2) for g_i in active_grads))
            
            self.cos_sim = (dot_product / (norm_dt * norm_g + 1e-20)).item()
            # -------------------------------
            # --- Raise learning rate based on cosine similarity ---
            # Increases lr if the similarity is positive (up to 2x if perfectly aligned)

            group['lr'] *=  1 + 0.005 * self.cos_sim  # Scale increase by cosine similarity
            

            group['lr'] = min(group['lr'], 0.05)
            # ------------------------------------------------------
            direction = 1 if group["maximize"] else -1
            for p_tensor, dt in zip(active_params, delta_theta):
                p_tensor.add_(dt, alpha=direction * group['lr'])


# class SR_CG(Optimizer):
#     def __init__(self, params, lr=0.001, cg_steps=10, init_reg=1, update_freq=1, warm_start=True, maximize=True):
#         defaults = dict(lr=lr, cg_steps=cg_steps, reg=init_reg, update_freq=update_freq, warm_start=warm_start, maximize=maximize, step=0)
#         super().__init__(params, defaults)
        
#         # Initialize state memory for warm starts
#         for group in self.param_groups:
#             for p in group['params']:
#                 self.state[p]['last_dt'] = torch.zeros_like(p.data)

#     @torch.no_grad()
#     def _fvp(self, p_list, v_chain, tanh_term, model, reg, params):
#         p_dict = {}
#         for p_tensor, param_ref in zip(p_list, params):
#             if param_ref is model.weight_matrix: p_dict['w'] = p_tensor
#             elif param_ref is model.vbias: p_dict['v'] = p_tensor
#             elif param_ref is model.hbias: p_dict['h'] = p_tensor
            
#         O_dot_p = ((v_chain @ p_dict['w']) * tanh_term).sum(dim=1) + \
#                   (v_chain @ p_dict['v']) + \
#                   (tanh_term @ p_dict['h'])
                  
#         O_dot_p_c = O_dot_p - O_dot_p.mean()
        
#         B = v_chain.size(0)
        
#         Sx_w = (v_chain.T @ (tanh_term * O_dot_p_c.unsqueeze(1))) / B
#         Sx_v = (v_chain.T @ O_dot_p_c) / B
#         Sx_h = (tanh_term.T @ O_dot_p_c) / B
        
#         Sx_list = []
#         for p_tensor, param_ref in zip(p_list, params):
#             if param_ref is model.weight_matrix:
#                 Sx_list.append(Sx_w + reg * p_tensor)
#             elif param_ref is model.vbias:
#                 Sx_list.append(Sx_v + reg * p_tensor)
#             elif param_ref is model.hbias:
#                 Sx_list.append(Sx_h + reg * p_tensor)
                
#         return Sx_list
    
#     @torch.no_grad()
#     def _get_adaptive_reg(self, v_chain, tanh_term, model, base_reg, scale):
#         B = v_chain.size(0)
#         v_sq_sum = (v_chain ** 2).sum(dim=1)
#         t_sq_sum = (tanh_term ** 2).sum(dim=1)
        
#         # Only W terms for the trace
#         mean_norm_s_k_sq = (v_sq_sum * t_sq_sum).mean()
#         mean_s_w = (v_chain.T @ tanh_term) / B
#         norm_mean_s_sq = (mean_s_w**2).sum()
        
#         trace_F = mean_norm_s_k_sq - norm_mean_s_sq
#         D = model.weight_matrix.numel() # Only dimension of W
        
#         adaptive_reg = base_reg * (trace_F ).item()/D

#         # adaptive_reg = base_reg 
#         return np.minimum(scale * adaptive_reg, 500.0)
    
#     @torch.no_grad()
#     def step(self, v_chain, model,scale=1, closure=None):
#         for group in self.param_groups:
#             group["step"] += 1
#             params = group["params"]
#             lr = group["lr"]
#             self.reg = np.minimum(scale*group["reg"],500.0)
#             M=float(model.weight_matrix.shape[0])
#             g = [p.grad.clone() for p in params]
            
#             # Lazy Preconditioning: Only run CG every 'update_freq' steps
#             if group["step"] % group["update_freq"] == 0 or group["step"] == 1:
                
#                 local_field = model.hbias + v_chain @ model.weight_matrix
#                 # tanh_term = torch.tanh(local_field)
#                 tanh_term = local_field/M
#                 # Clean, single call before the CG loop
#                 self.reg = self._get_adaptive_reg(v_chain, tanh_term, model, group["reg"], scale)
                
#                 if group["warm_start"] and group["step"] > 1:
#                     # Warm Start: Initialize with previous Natural Gradient
#                     delta_theta = [self.state[p]['last_dt'].clone() for p in params]
#                     S_dt = self._fvp(delta_theta, v_chain, tanh_term, model, self.reg, params)
#                     residual = [g_i - s_i for g_i, s_i in zip(g, S_dt)]
#                 else:
#                     # Cold Start
#                     delta_theta = [torch.zeros_like(p) for p in params]
#                     residual = [g_i.clone() for g_i in g]
                    
#                 p_vec = [r_i.clone() for r_i in residual]
#                 r_dot_r = sum(torch.sum(r * r) for r in residual)
                
#                 for _ in range(group["cg_steps"]):
#                     S_p = self._fvp(p_vec, v_chain, tanh_term, model, self.reg, params)
#                     p_Sp = sum(torch.sum(pv * spv) for pv, spv in zip(p_vec, S_p))
                    
#                     if p_Sp.item() <= 1e-8:
#                         break
                        
#                     alpha = (r_dot_r / p_Sp).item()
                    
#                     for dt, pv in zip(delta_theta, p_vec):
#                         dt.add_(pv, alpha=alpha)
                        
#                     for r, spv in zip(residual, S_p):
#                         r.sub_(spv, alpha=alpha)
                    
#                     new_r_dot_r = sum(torch.sum(r * r) for r in residual)
                    
#                     if new_r_dot_r.item() < 1e-6:
#                         break
                        
#                     beta = (new_r_dot_r / r_dot_r).item()
                    
#                     for pv, r in zip(p_vec, residual):
#                         pv.mul_(beta).add_(r)
                        
#                     r_dot_r = new_r_dot_r
                
#                 # Save the Natural Gradient for the next warm start
#                 for p_tensor, dt in zip(params, delta_theta):
#                     if p_tensor is model.weight_matrix:
#                         self.state[p_tensor]['last_dt'].copy_(dt)
#             else:
#                 # Fallback: Standard Gradient Update for intermediate steps
#                 delta_theta = g
#             # Apply the update
#             direction = 1 if group["maximize"] else -1
#             for p_tensor, dt in zip(params, delta_theta):
#                 #only if it is the weight_matrix
#                 if p_tensor is model.weight_matrix:
#                     p_tensor.add_(dt, alpha=direction * lr)

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
