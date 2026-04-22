from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler
import torch.nn.functional as F
@torch.jit.script 
def dlp_step(
    v: Tensor, 
    W: Tensor, 
    vb: Tensor, 
    hb: Tensor,
    beta: float, 
    alpha: float, 
    rnd_fw: Tensor, # Added for safe CUDA graph random proposals
    rnd_mh: Tensor, # Renamed for clarity
    states: Tensor, 
    scale: float, 
    shift: float, 
    dmala: bool
) -> Tensor:
    # 1. Forward Proposal
    # local_field_v = torch.matmul(v, W) + vb
    arg = hb + torch.matmul(v,W)
    # tanh_term = torch.tanh(arg)
    tanh_term = torch.sigmoid(arg)
    local_field_v = vb + torch.matmul(tanh_term,W.T)
    diff_fw = states - v.unsqueeze(-1)
    
    q_fw = torch.log_softmax(
        (0.5*beta * local_field_v.unsqueeze(-1) * diff_fw) - (0.5 * diff_fw.pow(2) / alpha),
        dim=2
    )

    p_plus1 = torch.exp(q_fw[:, :, 0])
    # Maps True/False to {1, -1} for Ising or {1, 0} for Bernoulli
    vp = scale * (rnd_fw < p_plus1).float() + shift
    
    if not dmala:
        # v.copy_(vp)
        return vp
        
    # 2. MH Correction
    idx_vp = (vp == shift).long().unsqueeze(-1)
    log_q_fw = q_fw.gather(2, idx_vp).squeeze(-1).sum(dim=1)

    # local_field_vp = torch.matmul(vp, W) + vb
    argp = hb + torch.matmul(vp,W)
    # tanh_termp = torch.tanh(argp)
    tanh_termp = torch.sigmoid(argp)
    local_field_vp = vb + torch.matmul(tanh_termp,W.T)

    diff_bw = states - vp.unsqueeze(-1)
    
    q_bw = torch.log_softmax(
        (0.5*beta * local_field_vp.unsqueeze(-1) * diff_bw) - (0.5 * diff_bw.pow(2) / alpha), 
        dim=2
    )
    
    idx_v = (v == shift).long().unsqueeze(-1)
    log_q_bw = q_bw.gather(2, idx_v).squeeze(-1).sum(dim=1)
    
    # energy_new = -torch.logaddexp(arg, -arg).sum(-1) - (v * vb).sum(-1)
    # energy_old = -torch.logaddexp(argp, -argp).sum(-1) - (vp * vb).sum(-1)
    energy_new = F.softplus(argp).sum(-1) + (vp * vb).sum(-1)
    energy_old = F.softplus(arg).sum(-1) + (v * vb).sum(-1)
    d_energy = energy_new - energy_old
    
    log_mh_ratio = beta*d_energy + log_q_bw - log_q_fw

    # print(torch.clip(log_mh_ratio.exp(),0,1).mean())
    accept = rnd_mh.log() < log_mh_ratio
    new_v = torch.where(accept.unsqueeze(-1), vp, v)
    # v.copy_(new_v)
    return new_v

import torch.nn.functional as F
# @torch.jit.script
def dlp_dmala_step(
    v: Tensor, W: Tensor, vb: Tensor, hb: Tensor,
    beta: Tensor, half_beta: Tensor, half_inv_alpha: Tensor, 
    rnd_fw: Tensor, rnd_mh: Tensor, 
    states: Tensor, scale: Tensor, shift: Tensor
) -> Tensor:
    # 1. Forward Proposal (DLP)
    arg = hb + torch.matmul(v, W)
    tanh_term = torch.sigmoid(arg)
    local_field_v = vb + torch.matmul(tanh_term, W.T)
    diff_fw = states - v.unsqueeze(-1)
    
    diff_fw_sq = diff_fw*diff_fw
    # Replaced 0.5 literals with precomputed GPU tensors
    q_fw = torch.log_softmax(
        (half_beta * local_field_v.unsqueeze(-1) * diff_fw) - (half_inv_alpha * diff_fw_sq),
        dim=2
    )
    p_plus1 = torch.exp(q_fw[:, :, 0])
    vp = scale * (rnd_fw < p_plus1).float() + shift
    
    # 2. MH Correction (DMALA)
    idx_vp = (vp == shift).long().unsqueeze(-1)
    log_q_fw = q_fw.gather(2, idx_vp).squeeze(-1).sum(dim=1)

    argp = hb + torch.matmul(vp, W)
    tanh_termp = torch.sigmoid(argp)
    local_field_vp = vb + torch.matmul(tanh_termp, W.T)

    diff_bw = states - vp.unsqueeze(-1)
    diff_bw_sq = diff_bw*diff_bw
    # Replaced 0.5 literals with precomputed GPU tensors
    q_bw = torch.log_softmax(
        (half_beta * local_field_vp.unsqueeze(-1) * diff_bw) - (half_inv_alpha * diff_bw_sq), 
        dim=2
    )
    idx_v = (v == shift).long().unsqueeze(-1)
    log_q_bw = q_bw.gather(2, idx_v).squeeze(-1).sum(dim=1)
    
    # Energy
    energy_new = F.softplus(argp).sum(-1) + (vp * vb).sum(-1)
    energy_old = F.softplus(arg).sum(-1) + (v * vb).sum(-1)
    
    # Log MH Ratio 
    log_mh_ratio = beta * (energy_new - energy_old) + log_q_bw - log_q_fw
    return torch.where(rnd_mh.log() < log_mh_ratio.unsqueeze(-1), vp, v)

class DLP(Sampler):
    def __init__(
        self,
        params: EBM,
        chains: dict[str, Tensor],
        num_steps: int,
        beta: float = 1,
        alpha:float = 0.6,
        dmala: bool = True,
        **kwargs,
    ):
        self.name = "DLP"
        self.chains = chains
        self.params = params
        self.beta = float(beta)
        self.alpha = float(alpha)
        # FIX: Force all scalars into GPU Tensors to prevent CPU-syncs during graph capture
        # self.beta = torch.tensor(beta, device=params.device)
        # self.alpha = torch.tensor(alpha, device=params.device)
        self.num_steps = num_steps
        self.dmala=dmala
        self.flags = []
        # Setup domain-specific variables (Ising vs Bernoulli)
        if self.params.visible_type == "ising":
            self.states = torch.tensor([1.0, -1.0], device=params.device).view(1, 2)
            self.scale = torch.tensor(2.0, device=params.device)
            self.shift = torch.tensor(-1.0, device=params.device)
        elif self.params.visible_type == "bernoulli":
            self.states = torch.tensor([1.0, 0.0], device=params.device).view(1, 2)
            self.scale = torch.tensor(1.0, device=params.device)
            self.shift = torch.tensor(0.0, device=params.device)
        else:
            raise ValueError(f"Unsupported visible_type: {self.params.visible_type}")

        # Placeholders for optional CUDA Graph execution
        self._graph = None
        self._static_v = None
        self._static_rnd = None
    
    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    @torch.no_grad()
    def sample(self, num_steps: int | None = None, use_cudagraph: bool = False, **kwargs):
        if num_steps is None:
            num_steps = self.num_steps

        v = self.chains['visible']

        if use_cudagraph:
            self._cudagraph_sample(v, num_steps)
            return

        # Standard Execution Path
        for _ in range(num_steps):
            # Generate noise eagerly per step to save VRAM
            rnd_fw = torch.rand((v.shape[0], v.shape[1]), device=v.device)
            rnd_mh = torch.rand((v.shape[0],), device=v.device)
            v = self.sample_step(v, rnd_fw, rnd_mh)
            
        self.chains['visible'] = v.clone()

    def sample_step(self, v: Tensor, rnd_fw: Tensor, rnd_mh: Tensor) -> Tensor:
        return dlp_step(
            v=v,
            W=self.params.weight_matrix,
            vb=self.params.vbias,
            hb=self.params.hbias,
            beta=self.beta,
            alpha=self.alpha,
            rnd_fw=rnd_fw,
            rnd_mh=rnd_mh,
            states=self.states,
            scale=self.scale,
            shift=self.shift,
            dmala=self.dmala
        )

    def _cudagraph_sample(self, v: torch.Tensor, num_steps: int):
        device = v.device
        
        W = self.params.weight_matrix.detach()
        vb, hb = self.params.vbias.detach(), self.params.hbias.detach()
        
        # Precompute constants as strict GPU tensors to avoid literal scalar copies during capture
        t_beta = torch.tensor(self.beta, device=device, dtype=torch.float32)
        t_half_beta = torch.tensor(0.5 * self.beta, device=device, dtype=torch.float32)
        t_half_inv_alpha = torch.tensor(0.5 / self.alpha, device=device, dtype=torch.float32)
        
        # Force states to the correct device (in case initialized on CPU)
        states = self.states.detach().to(device).detach()
        t_scale = torch.as_tensor(self.scale, device=device, dtype=torch.float32)
        t_shift = torch.as_tensor(self.shift, device=device, dtype=torch.float32)

        static_v = v.clone()
        static_rnd_fw = torch.empty((v.shape[0], v.shape[1]), device=device)
        static_rnd_mh = torch.empty((v.shape[0], 1), device=device) 
        
        capture_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(capture_stream):
            for _ in range(5): 
                static_rnd_fw.uniform_()
                static_rnd_mh.uniform_()
                _ = dlp_dmala_step(static_v, W, vb, hb, t_beta, t_half_beta, t_half_inv_alpha, 
                                   static_rnd_fw, static_rnd_mh, states, t_scale, t_shift)
        torch.cuda.synchronize() 
        
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            static_out = dlp_dmala_step(static_v, W, vb, hb, t_beta, t_half_beta, t_half_inv_alpha, 
                                        static_rnd_fw, static_rnd_mh, states, t_scale, t_shift)
            
        for _ in range(num_steps):
            static_rnd_fw.uniform_()
            static_rnd_mh.uniform_()
            g.replay()
            static_v.copy_(static_out)
            
        self.chains['visible'] = static_v.clone()

    @torch.compiler.disable
    def named_parameters(self):
        params_dict = self.params.named_parameters()
        params_dict["model_type"] = np.asarray(self.params.name, dtype="T")
        params_dict["sampler_type"] = np.asarray(self.name, dtype="T")
        match self.params.visible_type:
            case "bernoulli":
                chains_save = self.chains["visible"].bool().cpu().numpy()
            case "ising" | "categorical":
                chains_save = self.chains["visible"].to(torch.int16).cpu().numpy()
            case _:
                chains_save = self.chains["visible"].cpu().numpy()
        params_dict["parallel_chains"] = chains_save
        params_dict["beta"] = np.asarray(self.beta)
        params_dict["num_steps"] = np.asarray(self.num_steps)
        return params_dict

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        map_model: dict[str, type[EBM]],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PCD:
        names = ["model_type", "chains", "beta", "num_steps"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        model_type = str(named_params.pop("model_type"))
        chains_visible = torch.from_numpy(named_params.pop("parallel_chains")).to(
            device=device, dtype=dtype
        )
        beta = float(named_params.pop("beta"))
        num_steps = int(named_params.pop("num_steps"))
        # There should only remain the keys for the model loading
        params = map_model[model_type].set_named_parameters(
            named_params=named_params, device=device, dtype=dtype
        )
        chains = params.init_chains(chains_visible.shape[0], start_v=chains_visible)

        return PCD(params=params, chains=chains, num_steps=num_steps, beta=beta)

    def post_grad_update(self, params: EBM):
        self.params = params

    def get_metrics_display(self, metrics, **kwargs):
        return metrics

    def get_metrics_save(self):
        return None

    def pre_grad_update(self):
        pass
