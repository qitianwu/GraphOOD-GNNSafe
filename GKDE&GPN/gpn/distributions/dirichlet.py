import math
import torch
import torch.distributions as D
import gpn.distributions as E
from .base import ConjugatePrior

# pylint: disable=abstract-method
class Dirichlet(D.Dirichlet, ConjugatePrior):
    """
    Extension of PyTorch's native Dirichlet distribution to be used as a conjugate prior for the
    Categorical distribution.
    """

    @classmethod
    def __likelihood__(cls):
        return E.Categorical

    @property
    def mean_distribution(self):
        return E.Categorical(self.mean)

    def entropy(self):
        alpha = self.concentration
        k = alpha.size(-1)
        a0 = alpha.sum(-1)

        # Approximate for large a0
        t1 = 0.5 * (k - 1) + 0.5 * (k - 1) * math.log(2 * math.pi)
        t2 = 0.5 * alpha.log().sum(-1)
        t3 = (k - 0.5) * a0.log()
        approx = t1 + t2 - t3

        # Calculate exactly for lower a0
        t1 = alpha.lgamma().sum(-1) - a0.lgamma() - (k - a0) * a0.digamma()
        t2 = ((alpha - 1) * alpha.digamma()).sum(-1)
        exact = t1 - t2

        return torch.where(a0 >= 10000, approx, exact)

    @classmethod
    def from_sufficient_statistic(cls, sufficient_statistic, evidence, prior=None):
        if prior is not None:
            prior_sufficient_statistic, prior_evidence = prior
            assert prior_sufficient_statistic.size() == sufficient_statistic.size()[1:]
            assert prior_evidence.size() == (1,)
        else:
            prior_sufficient_statistic, prior_evidence = (0, 0)

        alpha = sufficient_statistic * evidence.unsqueeze(-1) + \
            prior_sufficient_statistic * prior_evidence
        return cls(alpha)

    def log_likeli_mean(self, data):
        alpha = self.concentration
        if alpha.dim() == 1:
            alpha = alpha.view(1, -1)

        a_sum = alpha.sum(-1)
        a_true = alpha.gather(-1, data.view(-1, 1)).squeeze(-1)
        return a_true.digamma() - a_sum.digamma()

    def to(self, *args, **kwargs):
        self.concentration = self.concentration.to(*args, **kwargs)
        return self
