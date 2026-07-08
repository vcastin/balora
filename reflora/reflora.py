import math
import torch

from torch import nn


def _get_lora_weights(model):
    """Extract (lora_A, lora_B) pairs from a PEFT model using duck typing,
    compatible with both peft and peft_ga LoRA layers."""
    # Unwrap PeftModel / LoraModel wrappers
    lora_model = model
    if hasattr(model, 'base_model'):
        lora_model = model.base_model

    lora_weights = []
    for module in lora_model.modules():
        # Duck-type check: any LoRA layer (peft or peft_ga) has lora_A
        if not hasattr(module, 'lora_A'):
            continue

        active_adapters = getattr(module, 'active_adapters', ['default'])
        for adapter in active_adapters:
            if adapter not in module.lora_A:
                continue
            lora_A = module.lora_A[adapter].weight  # shape: r x m
            lora_B = module.lora_B[adapter].weight  # shape: n x r
            lora_weights.append((lora_A, lora_B))

    return lora_weights


class Refactorer:
    def __init__(self,
                 model: nn.Module,
                 warmup_steps: int = 0,
                 re_init: bool = False,
                 interval: int = 1,
                 use_scalar: bool = False
                 ) -> None:
        self.warmup_steps = warmup_steps
        self.use_scalar = use_scalar
        self.interval = interval

        self.lora_weights = _get_lora_weights(model)

        if re_init:
            lora_model = model.base_model if hasattr(model, 'base_model') else model
            for module in lora_model.modules():
                if not hasattr(module, 'lora_A'):
                    continue
                active_adapters = getattr(module, 'active_adapters', ['default'])
                for adapter in active_adapters:
                    if adapter not in module.lora_A:
                        continue
                    lora_A = module.lora_A[adapter].weight
                    lora_B = module.lora_B[adapter].weight
                    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(lora_B, a=math.sqrt(5))
                    base_layer = module.get_base_layer()
                    dtype = base_layer.weight.dtype
                    fan_in_fan_out = getattr(module, 'fan_in_fan_out', False)
                    w = base_layer.weight.data.to(torch.float32)
                    if fan_in_fan_out:
                        w = w.t()
                    w -= module.scaling[adapter] * lora_B @ lora_A
                    if fan_in_fan_out:
                        w = w.t()
                    base_layer.weight.data = w.to(dtype)

    @torch.no_grad()
    def dummy_step(self) -> None:
        """Refactoring via gradient preconditioning (matrix mode, pre-step hook)."""
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return
        else:
            self.skip_steps = self.interval - 1

        for lora_A, lora_B in self.lora_weights:
            if lora_A.grad is None or lora_B.grad is None:
                continue

            if self.use_scalar:
                S = torch.linalg.norm(lora_B.data) / torch.linalg.norm(lora_A.data)
                Sinv = 1 / S
                lora_A.grad *= Sinv
                lora_B.grad *= S
            else:
                eps = torch.finfo(lora_A.dtype).eps
                sigmaA_sq, VA = torch.linalg.eigh(lora_A.data @ lora_A.data.t())
                sigmaA = torch.sqrt(sigmaA_sq)

                M_right = lora_B.data @ (VA * sigmaA)
                sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
                sigmaM = torch.sqrt(sigmaM_sq)

                S_left = (VA * (1 / (sigmaA + eps))) @ VM
                S = S_left * sigmaM @ S_left.t()

                Sinv_left = (VA * sigmaA) @ VM
                Sinv = (Sinv_left * (1 / (sigmaM + eps))) @ Sinv_left.t()

                lora_A.grad = Sinv @ lora_A.grad
                lora_B.grad @= S

    @torch.no_grad()
    def step(self, optimizer) -> None:
        """Real refactoring: reparameterize weights and adjust optimizer states (scalar mode, post-step hook)."""
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return
        else:
            self.skip_steps = self.interval - 1

        for lora_A, lora_B in self.lora_weights:
            if self.use_scalar:
                S = torch.linalg.norm(lora_B.data) / torch.linalg.norm(lora_A.data)
                P = torch.sqrt(S)

                lora_A.data *= P
                lora_B.data *= 1 / P

                if optimizer.state and lora_A in optimizer.state and 'exp_avg' in optimizer.state[lora_A]:
                    optimizer.state[lora_A]['exp_avg'] *= 1 / P
                    optimizer.state[lora_A]['exp_avg_sq'] *= 1 / S
                    optimizer.state[lora_B]['exp_avg'] *= P
                    optimizer.state[lora_B]['exp_avg_sq'] *= S
            else:
                eps = torch.finfo(lora_A.dtype).eps
                sigmaA_sq, VA = torch.linalg.eigh(lora_A.data @ lora_A.data.t())
                sigmaA = torch.sqrt(sigmaA_sq)

                M_right = lora_B.data @ (VA * sigmaA)
                sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
                sigmaM_sqrt = torch.pow(sigmaM_sq, 0.25)

                Pt = ((VA * (1 / (sigmaA + eps))) @ (VM * sigmaM_sqrt)).t()
                Ptinv = (VA * sigmaA) @ (VM * (1 / sigmaM_sqrt))

                lora_A.data = Pt @ lora_A.data
                lora_B.data @= Ptinv

    def integrate_into_optimizer(self, optimizer) -> None:
        self.skip_steps = self.warmup_steps

        if self.use_scalar:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.step(optimizer)
            optimizer.register_step_post_hook(refactor_hook)
            self.step(optimizer)  # initial call since we use a post-hook
        else:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.dummy_step()
            optimizer.register_step_pre_hook(refactor_hook)
