import torch

def orthogonal_init_lora(m):
    if not hasattr(m, "lora_A") or not hasattr(m, "lora_B"):
        return
    r, in_dim = m.lora_A.default.weight.shape
    out_dim, _ = m.lora_B.default.weight.shape
    with torch.no_grad():
        Qa, _ = torch.linalg.qr(torch.randn(in_dim, r))
        Qb, _ = torch.linalg.qr(torch.randn(out_dim, r))
        m.lora_A.default.weight.copy_(Qa.T)
        m.lora_B.default.weight.copy_(Qb)