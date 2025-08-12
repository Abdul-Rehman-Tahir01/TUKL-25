import torch
import torch.nn as nn


# =========== Module Wrapper for Linear Layer =========== 
class ModuleWrapperLinear(nn.Module):
    """Wrapper for nn.Module (Linear) with support for sample & n_samples propagation"""
    def __init__(self):
        super(ModuleWrapperLinear, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x, sample=True, n_samples=1):
        # x can be (batch, features) or (n_samples, batch, features)
        for module in self.children():
            name = module._get_name()
            # If module is a BBB layer, pass sample & n_samples
            if 'BBB' in name:
                x = module(x, sample=sample, n_samples=n_samples)
            else:
                # Non-BBB modules (activations, etc.)
                # If x has samples dim, merge before applying, then unmerge
                if x.dim() == 3:
                    n, b, f = x.shape
                    x = x.reshape(n * b, f)        # <-- changed .view -> .reshape
                    x = module(x)
                    # If module changes feature size, recompute new feature dim
                    new_f = x.shape[1]
                    x = x.reshape(n, b, new_f)    # <-- changed .view -> .reshape
                else:
                    x = module(x)
        # compute kl from all modules that have kl_loss (independent of n_samples)
        kl = torch.tensor(0.0, device=x.device)
        for module in self.modules():
            if hasattr(module, 'kl_loss') and module is not self:
                kl = kl + module.kl_loss()
        return x, kl
    


# =========== Module Wrapper for Conv1d Layer =========== 
class ModuleWrapperConv1d(nn.Module):
    """Wrapper for nn.Module (Conv1D) with support for sample & n_samples propagation"""
    def __init__(self):
        super(ModuleWrapperConv1d, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x, sample=True, n_samples=1):
        for module in self.children():
            name = module._get_name()
            if 'BBB' in name:
                x = module(x, sample=sample, n_samples=n_samples)
            else:
                if x.dim() == 4:
                    n, b, c, l = x.shape
                    x = x.reshape(n * b, c, l)
                    x = module(x)
                    new_c, new_l = x.shape[1], x.shape[2]
                    x = x.reshape(n, b, new_c, new_l)
                else:
                    x = module(x)
        kl = torch.tensor(0.0, device=x.device)
        for module in self.modules():
            if hasattr(module, 'kl_loss') and module is not self:
                kl = kl + module.kl_loss()
        return x, kl