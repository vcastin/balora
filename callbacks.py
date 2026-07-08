import torch
from transformers import TrainerCallback
import os
import json
from filelock import FileLock

class BestProjectionCallback(TrainerCallback):
    def __init__(self, model, project_every, num_iterations=10):
        self.model = model
        self.project_every = project_every
        self.num_iterations = num_iterations
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step +=1

        if self._step % self.project_every == 0:
            for module in self.model.modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A  # [r, in_features]
                    lora_B = module.lora_B  # [out_features, r]

                    A = lora_A.default.weight.data      # [r, in]
                    B = lora_B.default.weight.data      # [out, r]
                    
                    try:
                        U_A, D_A, V_A = torch.linalg.svd(A, full_matrices=False)
                    except RuntimeError:
                        print("SVD failed, skipping projection.")
                        continue
                    S_A, R_A = (U_A * D_A) @ U_A.T, U_A @ V_A

                    try:
                        U_B, D_B, V_B = torch.linalg.svd(B, full_matrices=False)
                    except RuntimeError:
                        print("SVD failed, skipping projection.")
                        continue
                    S_B, R_B = (V_B.T * D_B) @ V_B, U_B @ V_B

                    S = S_B @ S_A  # shape [r, r]
                    U, Sigma, V = torch.linalg.svd(S)
                    Sigma_sqrt = torch.sqrt(Sigma)

                    to_approx = 0.5 * Sigma_sqrt[:, None] * (U.T @ S_B + V @ S_A.T)
                    to_approx /= torch.linalg.norm(to_approx)

                    # newton-schulz approximation
                    O = to_approx
                    r = O.shape[0]
                    device = O.device
                    for _ in range(self.num_iterations):
                        O = 0.5 * O @ (3 * torch.eye(r).to(device) - O.T @ O)

                    S_B_proj = (U * Sigma_sqrt) @ O
                    S_A_proj = O.T @ (Sigma_sqrt[:, None] * V)

                    A_proj, B_proj = S_A_proj @ R_A, R_B @ S_B_proj

                    module.lora_A.default.weight.data = A_proj
                    module.lora_B.default.weight.data = B_proj


class ProjectionCallback(TrainerCallback):
    def __init__(self, model, project_every, start_step=0):
        self.model = model
        self.project_every = project_every
        self._step = 0
        self.start_step = start_step

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if state.global_step < self.start_step:
            return
        
        if self._step % self.project_every == 0:
            for module in self.model.modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_A = module.lora_A  # [r, in_features]
                    lora_B = module.lora_B  # [out_features, r]

                    A = lora_A.default.weight.data      # [r, in]
                    B = lora_B.default.weight.data      # [out, r]
                    
                    try:
                        U_A, D_A, V_A = torch.linalg.svd(A, full_matrices=False)
                    except RuntimeError:
                        print("SVD failed, skipping projection.")
                        continue
                    S_A, R_A = (U_A * D_A) @ U_A.T, U_A @ V_A

                    try:
                        U_B, D_B, V_B = torch.linalg.svd(B, full_matrices=False)
                    except RuntimeError:
                        print("SVD failed, skipping projection.")
                        continue
                    S_B, R_B = (V_B.T * D_B) @ V_B, U_B @ V_B

                    S = S_B @ S_A  # shape [r, r]
                    U, Sigma, V = torch.linalg.svd(S)
                    Sigma_sqrt = torch.sqrt(Sigma)

                    A_proj, B_proj = (Sigma_sqrt[:, None] * V) @ R_A, R_B @ (U * Sigma_sqrt)

                    module.lora_A.default.weight.data = A_proj
                    module.lora_B.default.weight.data = B_proj


class BalancingGapCallback(TrainerCallback):
    def __init__(self, model, balancing_gap_file="logs/balancing_gap.json", norm_file="logs/norms.json"):
        self.model = model
        self.balancing_gap_file = balancing_gap_file
        self.norm_file = norm_file
        self.norms = {}
        self.balancing_gaps = {}

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A  # [r, in_features]
                lora_B = module.lora_B  # [out_features, r]

                A = lora_A.default.weight.data      # [r, in]
                B = lora_B.default.weight.data      # [out, r]
                val = torch.linalg.norm(B.T @ B - A @ A.T).item()
                norm_A, norm_B = A.norm().item(), B.norm().item()

                if name not in self.balancing_gaps:
                    self.balancing_gaps[name] = []
                    self.norms[name + ".lora_A"] = []
                    self.norms[name + ".lora_B"] = []
                self.balancing_gaps[name].append(val)
                self.norms[name + ".lora_A"].append(norm_A)
                self.norms[name + ".lora_B"].append(norm_B)

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.balancing_gap_file, "w") as f:
            json.dump(self.balancing_gaps, f)
        print(f"Balancing gap saved to {self.balancing_gap_file}")
        with open(self.norm_file, "w") as f:
            json.dump(self.norms, f)
        print(f"Norms saved to {self.norm_file}")


class SaveValLossCallback(TrainerCallback):
    def __init__(self, output_file, trainer):
        self.output_file = output_file
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # run evaluation at epoch end manually
        metrics = self.trainer.evaluate()
        val_loss = metrics.get("eval_loss", None)
        if val_loss is None:
            return

        key = f"loss_after_epoch_{int(state.epoch)}"

        try:
            with open(self.output_file, "r") as f:
                data = json.load(f)
        except:
            data = {}

        data[key] = val_loss

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

        return