import torch
import torch.nn.functional as F
from torch.nn import Parameter

from utils.layer_wrappers import ModuleWrapperConv1d
from utils.kl_divergence_helper import KL_DIV


# ============ BayesianConv1d Layer ============
class BayesianConv1d(ModuleWrapperConv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, priors=None, device='cpu'):
        super(BayesianConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.device = device

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        # Parameters for weight posterior
        self.W_mu = Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size, device=self.device))
        self.W_rho = Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size, device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty(out_channels, device=self.device))
            self.bias_rho = Parameter(torch.empty(out_channels, device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def _compute_sigma(self):
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho)) if self.use_bias else None
        return W_sigma, bias_sigma

    def forward(self, input, sample=True, n_samples=1):
        W_sigma, bias_sigma = self._compute_sigma()
        self.W_sigma = W_sigma
        if self.use_bias:
            self.bias_sigma = bias_sigma

        # Case A: Multiple stochastic passes (vectorized)
        if sample and n_samples > 1:
            if input.dim() == 3:  # (batch, C, L)
                input = input.unsqueeze(0).expand(n_samples, *input.shape)
            else:
                if input.shape[0] != n_samples:
                    input = input[0].unsqueeze(0).expand(n_samples, *input.shape[1:])

            W_eps = torch.randn((n_samples, *self.W_mu.shape), device=self.device)
            weights = self.W_mu.unsqueeze(0) + W_sigma.unsqueeze(0) * W_eps

            outputs = []
            for i in range(n_samples):
                bias_i = None
                if self.use_bias:
                    bias_eps = torch.randn(self.bias_mu.shape, device=self.device)
                    bias_i = self.bias_mu + bias_sigma * bias_eps
                outputs.append(F.conv1d(input[i], weights[i], bias_i,
                                        stride=self.stride, padding=self.padding,
                                        dilation=self.dilation, groups=self.groups))
            return torch.stack(outputs, dim=0)

        # Case B: Single stochastic pass
        if sample and n_samples == 1:
            W_eps = torch.randn(self.W_mu.shape, device=self.device)
            weight = self.W_mu + W_sigma * W_eps
            bias = None
            if self.use_bias:
                bias_eps = torch.randn(self.bias_mu.shape, device=self.device)
                bias = self.bias_mu + bias_sigma * bias_eps
            return F.conv1d(input, weight, bias,
                            stride=self.stride, padding=self.padding,
                            dilation=self.dilation, groups=self.groups)

        # Case C: Deterministic
        return F.conv1d(input, self.W_mu, self.bias_mu if self.use_bias else None,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def kl_loss(self):
        if not hasattr(self, 'W_sigma'):
            self.W_sigma, _ = self._compute_sigma()
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            if not hasattr(self, 'bias_sigma'):
                _, self.bias_sigma = self._compute_sigma()
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl