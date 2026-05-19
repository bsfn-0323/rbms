import time

import numpy as np
import torch
from torch.optim import Optimizer
from rbms.optim import NGD
from tqdm.autonotebook import tqdm

from rbms.classes import EBM, Sampler
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import save_model, save_sampler
from rbms.training.utils import EarlyStopper


@torch.compile(dynamic=True, disable=True)
@torch.no_grad
def train(
    train_dataset: RBMDataset,
    test_dataset: RBMDataset,
    params: EBM,
    sampler: Sampler,
    optimizer: list[Optimizer],
    # early_stopper: EarlyStopper | None,
    batch_size: int,
    centered: bool,
    curr_update: int,
    pre_grad_update: torch.nn.Sequential,
    elapsed_time: float,
    checkpoints: np.ndarray,
    num_updates: int,
    filename: str,
    variational:bool,
    eta:float = 0.0,
):
    pbar = tqdm(
        initial=curr_update,
        total=num_updates,
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description(f"Training {params.name}")

    start = time.perf_counter()
    # initial_loss = None # Define this to capture the first real loss
    ema_loss = None     # Initialize as None to set on first iteration
    alpha=0.1
    ema_losses = []
    # skip_idx = 0
    # drop = 1.0
    # running_min = float('inf')
    
    for idx in range(curr_update + 1, num_updates + 1):
        
        for opt in optimizer:
            opt.zero_grad(set_to_none=False)
        #There should be an if logic for variational
        if variational:
            parallel_chains = sampler.get_conf_grad(batch=None)
            if getattr(train_dataset, 'K', None) is not None:
                loss = params.compute_hubbard_var_gradient(
                    K=train_dataset.K,
                    lam=train_dataset.lam,
                    chains=parallel_chains,
                    eta=eta,
                )
            else:
                j1, j2, j3 = train_dataset.J1, train_dataset.J2, train_dataset.J3
                loss = params.compute_var_gradient(
                    J1=j1,
                    J2=j2,
                    J3=j3,
                    chains=parallel_chains,
                    eta=eta,
                )
            if ema_loss is None:
                ema_loss = loss
                initial_loss = loss # Capture the starting plateau level

            ema_loss = alpha * loss + (1 - alpha) * ema_loss
            # 1. Detect the massive drop (compare to previous EMA before updating or use a threshold)
            # if skip_idx == 0 and np.abs(initial_loss - ema_loss)/initial_loss >0.9 : # Example: dropped by 50%
            #     skip_idx = idx
            #     print(f"Drop detected at: {idx}")

            # if skip_idx > 0:
            #     # 2. Update running minimum after the drop
            #     if ema_loss < running_min:
            #         running_min = ema_loss
                
            #     # 3. Detect overshoot (20% rise above the minimum reached after drop)
            #     if np.abs(ema_loss - running_min) / running_min > 0.5 and drop > 0.99999:
            #         drop = 0.9999
            #         print(f"Overshoot detected! Setting eta to zero at: {idx}")
            # eta = eta*drop
            ema_losses.append(ema_loss)

        else:
            # deltaE=None #Not needed in standard training
            loss=None
            batch = train_dataset.batch(batch_size)             
            data, weights = batch["data"], batch["weights"]     
            
            # Initialize batch
            curr_batch = params.init_chains(                    
                num_samples=data.shape[0],
                weights=weights,
                start_v=data,
            )
            parallel_chains = sampler.get_conf_grad(batch=data) 

            params.compute_gradient(
                data=curr_batch,
                chains=parallel_chains,
                centered=centered,
            )
        # Do a bunch of modification on the gradient

        pre_grad_update(None)
        params.pre_grad_update()
        sampler.pre_grad_update()

        for opt in optimizer:
            if isinstance(opt, NGD):
                opt.prepare(parallel_chains["visible"],scale =1)
            opt.step()

        params.post_grad_update()
        sampler.post_grad_update(params=params)

        # Get flags for save
        flags = []
        flags = params.save_flags(flags)
        flags = sampler.save_flags(flags)
        if idx in checkpoints or idx == num_updates:
            flags.append("checkpoint")

        if len(flags) > 0:
            names_params = (
                list(params.named_parameters().keys()) if len(optimizer) > 1 else ["all"]
            )
            learning_rates = np.asarray([opt.param_groups[0]["lr"] for opt in optimizer])

            metrics = {}
            # metrics = sampler.get_metrics_display(
            #     metrics, train_dataset=train_dataset, test_dataset=test_dataset,deltaE=deltaE
            # )
            metrics = sampler.get_metrics_display(
                metrics, train_dataset=train_dataset, test_dataset=test_dataset
            )
            pbar.write(f"=========== Update {idx} ===========")
            for k, v in metrics.items():
                pbar.write(f"{k}: {v}")
            pbar.write("learning rate :")
            for i in range(len(optimizer)):
                pbar.write(f"    - {names_params[i]} : {learning_rates[i]:.6f}")
            pbar.write(f"scale : {opt.reg:.3g}")
            pbar.write(f"cg steps: {opt.cg_step}")

            if variational:
                pbar.write("loss : ")
                pbar.write(f"{loss:.4f}")
            # pbar.write(metrics)
            curr_time = time.perf_counter() - start
            learning_rate = torch.tensor([opt.param_groups[0]["lr"] for opt in optimizer])
            save_model(
                filename=filename,
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                learning_rate=learning_rate,
                flags=flags,
                loss=loss,
                
            )

            save_sampler(filename, sampler, idx)
        pbar.update(1)
