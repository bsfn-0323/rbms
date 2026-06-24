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
    """Natural Gradient Descent via conjugate-gradient Fisher-vector products.

    Solves (F + λI) δ = g at each step, where F is the empirical Fisher matrix
    estimated from the current model chains.

    Usage::

        opt = NGD(model.parameters(), model=model, lr=1e-3)
        ...
        opt.prepare(v_chain)   # supply chain samples before each step
        opt.step()
    """

    def __init__(
        self,
        params,
        model: EBM,
        lr: float = 0.001,
        cg_steps: int = 20,
        init_reg: float = 1.0,
        update_freq: int = 1,
        warm_start: bool = False,
        maximize: bool = True,
        update_biases: bool = True,
    ):
        defaults = dict(
            lr=lr,
            cg_steps=cg_steps,
            reg=init_reg,
            update_freq=update_freq,
            warm_start=warm_start,
            maximize=maximize,
            update_biases=update_biases,
            step=0,
        )
        super().__init__(params, defaults)
        self.model = model
        self._v_chain: Tensor | None = None
        self._scale: float = 1.0
        # Diagnostic attributes, readable after each step
        self.reg: float = 0.0
        self.cos_sim: float = 1.0
        self.cg_step: int = 0

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["last_dt"] = torch.zeros_like(p.data)

    def prepare(self, v_chain: Tensor, scale: float = 1.0) -> None:
        """Set chain samples and scale for the next step()."""
        self._v_chain = v_chain
        self._scale = scale

    @torch.no_grad()
    def _adaptive_reg(self, v_eff: Tensor, tau: Tensor, base_reg: float) -> float:
        """Return λ = base_reg * trace(F_W) / D, clamped to be strictly positive.

        trace(F_W) = E[‖v‖²‖τ‖²] − ‖E[v τᵀ]‖²_F
        """
        B = v_eff.size(0)
        v_flat = v_eff.view(B, -1)

        mean_sq = (v_flat.square().sum(1) * tau.square().sum(1)).mean()
        mean_outer = (v_flat.T @ tau) / B
        trace_F = mean_sq - mean_outer.square().sum()

        D = self.model.weight_matrix.numel()
        return max(1e-8, base_reg * trace_F.item()/D)

    @torch.no_grad()
    def _fvp(
        self,
        p_list: list[Tensor],
        v_eff: Tensor,
        tau: Tensor,
        reg: float,
        update_biases: bool,
    ) -> list[Tensor]:
        """Compute (F + reg·I) @ p via efficient batch outer products.

        F is the Fisher of the RBM model distribution with sufficient statistics
        O(v) = [v ⊗ τ, v, τ] where τ = σ(W^T v + b_h).
        """
        B = v_eff.size(0)
        v_flat = v_eff.view(B, -1)

        p_w = p_list[0]
        p_w_flat = p_w.view(-1, p_w.size(-1))

        # O(v)^T p per sample — shape (B,)
        O_dot_p = (v_flat @ p_w_flat * tau).sum(1)
        if update_biases:
            p_v, p_h = p_list[1], p_list[2]
            O_dot_p = O_dot_p + v_flat @ p_v.view(-1) + tau @ p_h

        # Center to compute the covariance form: Cov[O] = E[OO^T] − E[O]E[O]^T
        O_dot_p = O_dot_p - O_dot_p.mean()

        Fv_w = (v_flat.T @ (tau * O_dot_p.unsqueeze(1))) / B
        result = [Fv_w.view_as(p_w) + reg * p_w]

        if update_biases:
            result.append((v_flat.T @ O_dot_p).view_as(p_v) / B + reg * p_v)
            result.append((tau.T @ O_dot_p) / B + reg * p_h)

        return result

    @torch.no_grad()
    def step(self, closure=None):
        max_lr = 1/math.sqrt(self.model.weight_matrix.numel()) # Cap learning rate to prevent divergence on large models
        end = None
        if closure is not None:
            with torch.enable_grad():
                end = closure()

        if self._v_chain is None:
            raise RuntimeError("Call prepare(v_chain) before step().")

        v_chain = self._v_chain
        scale = self._scale

        for group in self.param_groups:
            group["step"] += 1
            params = group["params"]
            update_biases = group["update_biases"]
            g = [p.grad.clone() for p in params]

            run_cg = (group["step"] == 1) or (group["step"] % group["update_freq"] == 0)

            if run_cg:
                grad_v, grad_h, _ = self.model.compute_energy_visible_gradient(v_chain)
                v_eff = -grad_v
                tau = -grad_h
                self.reg = scale * self._adaptive_reg(v_eff, tau, group["reg"])

                if update_biases:
                    active_params, active_grads = params, g
                else:
                    active_params = [p for p in params if p is self.model.weight_matrix]
                    active_grads = [p.grad.clone() for p in active_params]

                # Warm start: initialise CG from the previous natural gradient
                if group["warm_start"] and group["step"] > 1:
                    delta = [self.state[p]["last_dt"].clone() for p in active_params]
                    residual = [
                        g_i - s_i
                        for g_i, s_i in zip(
                            active_grads,
                            self._fvp(delta, v_eff, tau, self.reg, update_biases),
                        )
                    ]
                else:
                    delta = [torch.zeros_like(p) for p in active_params]
                    residual = [g_i.clone() for g_i in active_grads]

                p_vec = [r.clone() for r in residual]
                r_sq = sum(r.square().sum() for r in residual)
                init_r_norm = r_sq.sqrt().item() or 1e-20
                current_r_norm = init_r_norm
                self.cg_step = 0

                while current_r_norm / init_r_norm > 0.01:
                    if self.cg_step >= group["cg_steps"]:
                        break
                # for _ in range(group["cg_steps"]):
                    Sp = self._fvp(p_vec, v_eff, tau, self.reg, update_biases)
                    p_Sp = sum((pv * spv).sum() for pv, spv in zip(p_vec, Sp))
                    if p_Sp.item() <= 1e-20:
                        break

                    alpha = (r_sq / p_Sp).item()
                    for d, pv in zip(delta, p_vec):
                        d.add_(pv, alpha=alpha)
                    for r, spv in zip(residual, Sp):
                        r.sub_(spv, alpha=alpha)

                    new_r_sq = sum(r.square().sum() for r in residual)
                    current_r_norm = new_r_sq.sqrt().item()
                    if new_r_sq.item() < 1e-20:
                        break

                    beta = (new_r_sq / r_sq).item()
                    for pv, r in zip(p_vec, residual):
                        pv.mul_(beta).add_(r)

                    r_sq = new_r_sq
                    self.cg_step += 1

                for p_tensor, d in zip(active_params, delta):
                    self.state[p_tensor]["last_dt"].copy_(d)

            else:
                # Lazy step: apply plain gradient for skipped CG iterations
                active_params = (
                    params if update_biases
                    else [p for p in params if p is self.model.weight_matrix]
                )
                active_grads = [p.grad.clone() for p in active_params]
                delta = active_grads

            # Cosine similarity between natural gradient and raw gradient
            # dot = sum((d * g_i).sum() for d, g_i in zip(delta, active_grads))
            # norm_d = sum(d.square().sum() for d in delta).sqrt()
            # norm_g = sum(g_i.square().sum() for g_i in active_grads).sqrt()
            # self.cos_sim = (dot / (norm_d * norm_g + 1e-20)).item()

            # group["lr"] *=1.0 + 0.00075*self.cos_sim # Scale learning rate by cosine similarity
            # group["lr"] = min(group["lr"], max_lr)
            sign = 1 if group["maximize"] else -1
            for p_tensor, d in zip(active_params, delta):
                p_tensor.add_(d, alpha=sign * group["lr"])

        return end


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
    if args["optim"] == "ngd":
        if not isinstance(learning_rate, Tensor):
            learning_rate = torch.tensor([learning_rate])
        optimizer = [
            NGD(
                params.parameters(),
                model=params,
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
                    **({"max_lr": max_lr} if optim_class is SGD_cossim else {}),
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
