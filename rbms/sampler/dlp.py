from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler

@torch.jit.script
def dlp_step(
    v: Tensor, 
    W: Tensor, 
    vb: Tensor, 
    hb: Tensor,
    beta: float, 
    alpha: float, 
    rnd: Tensor, 
    states: Tensor, 
    scale: float, 
    shift: float, 
    dmala: bool
) -> Tensor:
    # 1. Forward Proposal
    # local_field_v = torch.matmul(v, W) + vb
    tanh_term = torch.tanh(hb + torch.matmul(v,W))
    local_field_v = vb + torch.matmul(tanh_term,W.T)
    diff_fw = states - v.unsqueeze(-1)
    
    q_fw = torch.log_softmax(
        (beta * local_field_v.unsqueeze(-1) * diff_fw) - (0.5 * diff_fw.pow(2) / alpha),
        dim=2
    )

    p_plus1 = torch.exp(q_fw[:, :, 0])
    # Maps True/False to {1, -1} for Ising or {1, 0} for Bernoulli
    vp = scale * (torch.rand_like(p_plus1) < p_plus1).float() + shift
    
    if not dmala:
        return vp
        
    # 2. MH Correction
    idx_vp = (vp == shift).long().unsqueeze(-1)
    log_q_fw = q_fw.gather(2, idx_vp).squeeze(-1).sum(dim=1)

    # local_field_vp = torch.matmul(vp, W) + vb

    tanh_termp = torch.tanh(hb + torch.matmul(vp,W))
    local_field_vp = vb + torch.matmul(tanh_term,W.T)

    diff_bw = states - vp.unsqueeze(-1)
    
    q_bw = torch.log_softmax(
        (beta * local_field_vp.unsqueeze(-1) * diff_bw) - (0.5 * diff_bw.pow(2) / alpha), 
        dim=2
    )
    
    idx_v = (v == shift).long().unsqueeze(-1)
    log_q_bw = q_bw.gather(2, idx_v).squeeze(-1).sum(dim=1)
    
    energy_new = beta * (local_field_vp * vp).sum(dim=1)
    energy_old = beta * (local_field_v * v).sum(dim=1)
    d_energy = energy_new - energy_old
    
    log_mh_ratio = -d_energy + log_q_bw - log_q_fw
    accept = torch.log(rnd) < log_mh_ratio
    
    return torch.where(accept.unsqueeze(-1), vp, v)

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
        self.beta = beta
        self.alpha = alpha
        self.num_steps = num_steps
        self.dmala=dmala
        self.flags = []
        # Setup domain-specific variables (Ising vs Bernoulli)
        if self.params.visible_type == "ising":
            self.states = torch.tensor([1.0, -1.0], device=params.device).view(1, 2)
            self.scale = 2.0
            self.shift = -1.0
        elif self.params.visible_type == "bernoulli":
            # Assuming Bernoulli is {1, 0} to keep index 0 as probability of 1
            self.states = torch.tensor([1.0, 0.0], device=params.device).view(1, 2)
            self.scale = 1.0
            self.shift = 0.0
        else:
            raise ValueError(f"Unsupported visible_type: {self.params.visible_type}")

        # Placeholders for optional CUDA Graph execution
        self._graph = None
        self._static_v = None
        self._static_rnd = None
    
    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    # def sample(self, num_steps: int | None, **kwargs):
    #     v = self.chains['visible']
    #     rnds = torch.randn((v.shape[0],num_steps),device = params.device)
    #     for i in range(num_steps):
    #         v = self.sample_step(v,rnds[:,i])
    def sample(self, num_steps: int | None, use_cudagraph: bool = False, **kwargs):
        if num_steps is None:
            num_steps = self.num_steps

        v = self.chains['visible']
        rnds = torch.rand((num_steps, v.shape[0]), device=self.params.device) # rand for log(rnd) comparison
        
        # Standard execution using JIT
        if not use_cudagraph:
            for i in range(num_steps):
                v = self.sample_step(v, rnds[i])
            self.chains['visible'] = v
            return
            
        # Optional CUDA Graph execution path for extreme profiling
        self._cudagraph_sample(v, rnds, num_steps)

    def sample_step(self, v: Tensor, rnd: Tensor) -> Tensor:
        return dlp_step(
            v=v,
            W=self.params.weight_matrix,
            vb=self.params.vbias,
            hb=self.params.hbias,
            beta=self.beta,
            alpha=self.alpha,
            rnd=rnd,
            states=self.states,
            scale=self.scale,
            shift=self.shift,
            dmala=self.dmala
        )

    def _cudagraph_sample(self, v: Tensor, rnds: Tensor, num_steps: int):
        """Captures and replays the step using CUDA Graphs for zero overhead."""
        if self._graph is None:
            # 1. Initialize static buffers
            self._static_v = v.clone()
            self._static_rnd = torch.rand_like(rnds[0])
            
            # 2. Warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._static_v = self.sample_step(self._static_v, self._static_rnd)
            torch.cuda.current_stream().wait_stream(s)
            
            # 3. Capture
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_v = self.sample_step(self._static_v, self._static_rnd)
        
        # Ensure input buffer matches current chain state
        self._static_v.copy_(v)
        
        # 4. Replay
        for i in range(num_steps):
            self._static_rnd.copy_(rnds[i])
            self._graph.replay()
            
        self.chains['visible'] = self._static_v.clone()
    # def sample_step(self,v,rnd):
    #     # v = self.chains['visible']
    #     local_field_v = v@params.weight_matrix + params.vbias
    #     diff_fw = states - v.unsqueeze(-1)
    #     q_fw = torch.log_softmax(
    #         self.beta*local_field_v.unsqueeze(-1) * diff_fw - 0.5 * diff_fw.pow(2) / alpha,
    #         dim = 2
    #     )

    #     p_plus1 = q_fw[:,:,0].exp()
    #     vp = 2.0 * (torch.rand_like(p_plus1) < p_plus1).float() - 1.0 #change
        
    #     if not self.dmala:
    #         return vp
    #     else:
    #         idx_xp = (xp == -1.0).long().unsqueeze(-1) #change
    #         log_q_fw = q_fw.gather(2, idx_xp).squeeze(-1).sum(dim=1)

    #         local_field_vp =  vp@params.weight_matrix + params.vbias
    #         diff_bw = states - vp.unsqueeze(-1)
    #         q_bw = torch.log_softmax(
    #             (beta * local_field_vp.unsqueeze(-1) * diff_bw) - (0.5 * diff_bw.pow(2) / alpha), 
    #             dim=2
    #         )
    #         idx_x = (x == -1.0).long().unsqueeze(-1) #change
    #         log_q_bw = q_bw.gather(2, idx_x).squeeze(-1).sum(dim=1) # Shape: (B,)
            
    #         # Batch energy difference ΔE = H(xp) - H(x)
    #         # Vectorized dot product using element-wise mul + sum: (B,)
    #         energy_new = beta * (local_field_xp * xp).sum(dim=1)
    #         energy_old = beta * (local_field_x * x).sum(dim=1)
    #         d_energy = energy_new - energy_old
            
    #         # log(α) = -ΔE + log(q_rev) - log(q_fwd)
    #         log_mh_ratio = -d_energy + log_q_bw - log_q_fw
            
    #         accept = torch.log(rnd) < log_mh_ratio # Shape: (B,)
    #         return torch.where(accept.unsqueeze(-1),vp, v)

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
