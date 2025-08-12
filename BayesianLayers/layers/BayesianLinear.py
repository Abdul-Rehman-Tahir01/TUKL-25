import torch
import torch.nn.functional as F
from torch.nn import Parameter

from utils.layer_wrappers import ModuleWrapperLinear
from utils.kl_divergence_helper import KL_DIV


# ============ BayesianLinear Layer ============
class BayesianLinear(ModuleWrapperLinear):
    def __init__(self, in_features, out_features, bias=True, priors=None, device='cpu'):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

        self.W_mu  = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu  = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
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
        # stable softplus
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        else:
            bias_sigma = None
        return W_sigma, bias_sigma

    def forward(self, input, sample=True, n_samples=1):
        """
        input: (batch, in_features)  OR  (n_samples, batch, in_features)
        returns:
            if n_samples>1: outputs (n_samples, batch, out_features)
            else: outputs (batch, out_features)
        """
        W_sigma, bias_sigma = self._compute_sigma()
        self.W_sigma = W_sigma            # keep attributes for kl_loss
        if self.use_bias:
            self.bias_sigma = bias_sigma

        # Case A: sample=True and n_samples>1 -> vectorized multiple draws
        if sample and n_samples > 1:
            if input.dim() == 2:
                batch_in = input.unsqueeze(0).expand(n_samples, input.size(0), input.size(1))  # (n, b, i)
            else:
                # input already shaped (n_input, b, i)
                batch_in = input
                if batch_in.shape[0] != n_samples:
                    batch_in = batch_in[0].unsqueeze(0).expand(n_samples, *batch_in.shape[1:])

            # sample weights for all n at once
            W_eps = torch.randn((n_samples, *self.W_mu.size()), device=self.device)  # (n, out, in)
            weights = self.W_mu.unsqueeze(0) + W_sigma.unsqueeze(0) * W_eps             # (n, out, in)

            # batch_in: (n, b, in); weights: (n, out, in)
            outputs = torch.einsum('noi,nbi->nbo', weights, batch_in)

            if self.use_bias:
                bias_eps = torch.randn((n_samples, *self.bias_mu.size()), device=self.device)  # (n, out)
                biases = self.bias_mu.unsqueeze(0) + bias_sigma.unsqueeze(0) * bias_eps       # (n, out)
                outputs = outputs + biases.unsqueeze(1)  # (n, b, out)
            return outputs

        # Case B: sample=True and n_samples == 1 -> single stochastic draw
        if sample and n_samples == 1:
            W_eps = torch.randn(self.W_mu.size(), device=self.device)
            weight = self.W_mu + W_eps * W_sigma
            if self.use_bias:
                bias_eps = torch.randn(self.bias_mu.size(), device=self.device)
                bias = self.bias_mu + bias_eps * bias_sigma
            else:
                bias = None
            # input must be (batch, in) or (n, b, f)
            if input.dim() == 3:
                n, b, f = input.shape
                flat = input.reshape(n * b, f)                # <-- reshape
                out_flat = F.linear(flat, weight, bias)
                out = out_flat.reshape(n, b, -1)              # <-- reshape
                return out
            else:
                return F.linear(input, weight, bias)

        # Case C: deterministic (use mu)
        weight = self.W_mu
        if self.use_bias:
            bias = self.bias_mu
        else:
            bias = None
        if input.dim() == 3:
            n, b, f = input.shape
            flat = input.reshape(n * b, f)                    # <-- reshape
            out_flat = F.linear(flat, weight, bias)
            out = out_flat.reshape(n, b, -1)                  # <-- reshape
            return out
        else:
            return F.linear(input, weight, bias)

    def kl_loss(self):
        # ensure W_sigma and bias_sigma exist
        if not hasattr(self, 'W_sigma'):
            self.W_sigma, _ = self._compute_sigma()
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            if not hasattr(self, 'bias_sigma'):
                _, self.bias_sigma = self._compute_sigma()
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl