import torch
from torch import nn
from torch import Tensor
import torch.distributions as tdist


class MixtureDensity(nn.Module):
    """layer representing a density from a mixture of gaussians"""

    def __init__(self, dim: int, n_components: int = 20, mixture_type: str = 'normal_mixture'):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.mixture_type = mixture_type

        if self.mixture_type == 'normal_mixture':
            # Isotropic Gaussians
            self.log_pi = torch.nn.Parameter(torch.randn(self.n_components, 1))
            self.log_pi.requires_grad = True
            self.mu = torch.nn.Parameter(torch.randn(self.n_components, self.dim))
            self.mu.requires_grad = True
            self.log_sigma_ = torch.nn.Parameter(torch.randn(self.n_components, self.dim, self.dim)/100.)
            self.log_sigma_.requires_grad = True

            self.softmax = nn.Softmax(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        pi = self.softmax(self.log_pi)
        # Parametrization with LL^T where diagonal of L are positives.
        sigma = self.log_sigma_ * torch.tril(torch.ones_like(self.log_sigma_))
        sigma = sigma - torch.diag_embed(torch.diagonal(sigma, dim1=-2, dim2=-1)) + torch.diag_embed(torch.diagonal(torch.exp(sigma), dim1=-2, dim2=-1)) + .001 * torch.diag_embed(torch.ones(self.n_components, self.dim)).to(x.device.type)
        dist = tdist.MultivariateNormal(loc=self.mu, scale_tril=sigma)

        expand_x = x.unsqueeze(1).repeat(1, self.n_components, 1)
        p_expand_x = torch.exp(dist.log_prob(expand_x))
        log_prob_x = torch.log(torch.matmul(p_expand_x, pi).squeeze() + 1e-10)
        return log_prob_x

    def log_prob(self, x: Tensor) -> Tensor:
        return self.forward(x)
