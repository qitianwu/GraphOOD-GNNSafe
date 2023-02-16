import torch.distributions as D
import gpn.distributions as E
from .base import Likelihood

# pylint: disable=abstract-method
class Categorical(D.Categorical, Likelihood):
    """
    Extension of PyTorch's native Categorical distribution to be used as a likelihood function.
    """

    @classmethod
    def __prior__(cls):
        return E.Dirichlet

    @classmethod
    def from_model_params(cls, x):
        return cls(x.softmax(-1))

    @property
    def mean(self):
        return self.logits.argmax(-1)

    @property
    def sufficient_statistic_mean(self):
        return self.probs

    def to(self, *args, **kwargs):
        if 'probs' in self.__dict__:
            self.probs = self.probs.to(*args, **kwargs)
        else:
            self.logits = self.logits.to(*args, **kwargs)
        return self
