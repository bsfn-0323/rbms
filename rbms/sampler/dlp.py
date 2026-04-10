from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler


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
        self.flags = []
        #check if it is BBRBM or IIRBM
        if self.params.visible_type == "ising":
            self.states = torch.tensor([1.0, -1.0], device=params.device).view(1, 2)
        
        if self.params.visible_type == "bernoulli":
            self.states = torch.tensor([1.0, -1.0], device=params.device).view(1, 2)
    
    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    def sample(self, num_steps: int | None, **kwargs):
        v = self.chains['visible']
        rnds = torch.randn((v.shape[0],num_steps),device = params.device)
        for i in range(num_steps):
            v = self.sample_step(v,rnds[:,i])

    def sample_step(self,v,rnd):
        # v = self.chains['visible']
        local_field_v = v@params.weight_matrix + params.vbias
        diff_fw = states - v.unsqueeze(-1)
        q_fw = torch.log_softmax(
            self.beta*local_field_v.unsqueeze(-1) * diff_fw - 0.5 * diff_fw.pow(2) / alpha,
            dim = 2
        )

        p_plus1 = q_fw[:,:,0].exp()
        vp = 2.0 * (torch.rand_like(p_plus1) < p_plus1).float() - 1.0 #change
        
        if not self.dmala:
            return vp
        else:
            idx_xp = (xp == -1.0).long().unsqueeze(-1) #change
            log_q_fw = q_fw.gather(2, idx_xp).squeeze(-1).sum(dim=1)

            local_field_vp =  vp@params.weight_matrix + params.vbias
            diff_bw = states - vp.unsqueeze(-1)
            q_bw = torch.log_softmax(
                (beta * local_field_vp.unsqueeze(-1) * diff_bw) - (0.5 * diff_bw.pow(2) / alpha), 
                dim=2
            )
            idx_x = (x == -1.0).long().unsqueeze(-1) #change
            log_q_bw = q_bw.gather(2, idx_x).squeeze(-1).sum(dim=1) # Shape: (B,)
            
            # Batch energy difference ΔE = H(xp) - H(x)
            # Vectorized dot product using element-wise mul + sum: (B,)
            energy_new = beta * (local_field_xp * xp).sum(dim=1)
            energy_old = beta * (local_field_x * x).sum(dim=1)
            d_energy = energy_new - energy_old
            
            # log(α) = -ΔE + log(q_rev) - log(q_fwd)
            log_mh_ratio = -d_energy + log_q_bw - log_q_fw
            
            accept = torch.log(rnd) < log_mh_ratio # Shape: (B,)
            return torch.where(accept.unsqueeze(-1),vp, v)

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
