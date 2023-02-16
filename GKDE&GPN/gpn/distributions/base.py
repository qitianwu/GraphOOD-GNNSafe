import torch
import torch.distributions as D

class ExponentialFamily(D.ExponentialFamily):
    """
    Shared base distribution for exponential family distributions.
    """

    @property
    def is_sparse(self):
        """
        Whether the distribution's parameters are sparse. Just returns `False`.
        """
        return False

    def is_contiguous(self):
        """
        Whether this distribution's parameters are contiguous. Just returns `True`.
        """
        return True

    def to(self, *args, **kwargs):
        """
        Moves the probability distribution to the specified device.
        """
        raise NotImplementedError

#--------------------------------------------------------------------------------------------------

class Likelihood(ExponentialFamily):
    """
    A likelihood represents a target distribution which has a conjugate prior. Examples are the
    Normal distribution for regression and the Categorical distribution for classification.

    Besides this class's abstract methods, a likelihood distribution must (at least) implement the
    methods/properties :code:`mean`, :code:`entropy` and :code:`log_prob`.
    """

    @classmethod
    def __prior__(cls):
        """
        The distribution class that the prior is based on.
        """
        raise NotImplementedError

    @classmethod
    def from_model_params(cls, x):
        """
        Returns the distribution as parametrized by some model. Although this is model-dependent,
        the model typically returns outputs on the real line and this method ensures that the
        parameters are valid (e.g. Softmax function over logits).

        Parameters
        ----------
        x: torch.Tensor [N, ...]
            The parameters of the distribution.

        Returns
        -------
        evidence.distributions.Likelihood
            The likelihood.
        """
        raise NotImplementedError

    @property
    def sufficient_statistic_mean(self):
        """
        Returns the mean (expectation) of the sufficient statistic of this distribution. That is,
        it returns the average of the sufficient statistic if infinitely many samples were drawn
        from this distribution.
        """
        raise NotImplementedError

    def uncertainty(self):
        """
        Returns some measure of uncertainty of the distribution. Usually, this is the entropy but
        distributions may choose to implement it differently if the entropy is intractable.
        """
        return self.entropy()

#--------------------------------------------------------------------------------------------------

class ConjugatePrior(ExponentialFamily):
    """
    A conjugate prior is an exponential family distribution which is conjugate for another
    (exponential family) distribution that is the underlying distribution for some likelihood
    function. The class of this underlying distribution must be available via the
    :code:`__likelihood__` property.

    Besides this class's abstract methods, a conjugate prior must (at least) implement the methods/
    properties :code:`mean` and :code:`entropy`.
    """

    @classmethod
    def __likelihood__(cls):
        """
        The distribution class that the likelihood function is based on.
        """
        raise NotImplementedError

    @classmethod
    def from_sufficient_statistic(cls, sufficient_statistic, evidence, prior=None):
        """
        Initializes this conjugate prior where parameters are computed from the given sufficient
        statistic and the evidence.

        Parameters
        ----------
        sufficient_statistic: torch.Tensor [N, ...]
            The sufficient statistic for arbitrarily many likelihood distributions (number of
            distributions N).
        evidence: torch.Tensor [N]
            The evidence for all likelihood distributions (i.e. the "degree of confidence").
        prior: tuple of (torch.Tensor[...], torch.Tensor [1]), default: None
            Optional prior to set on the sufficient statistic and the evidence. There always exists
            a bijective mapping between these priors and priors on the distribution's parameters.

        Returns
        -------
        Self
            An instance of this class.
        """
        raise NotImplementedError

    def log_likeli_mean(self, data):
        """
        Computes the mean (expectation) of the log-probability of observing the given data. The data
        is assumed to be distributed according to this prior's likelihood distribution.

        Parameters
        ----------
        data: torch.Tensor [N, ...]
            The observed values in the support of the likelihood distribution. The number of
            observations must be equal to the batch shape of this distribution (number of
            observations N).

        Returns
        -------
        torch.Tensor [N]
            The expectation of the log-probability for all observed values.
        """
        raise NotImplementedError

    @property
    def predictive_distribution(self):
        """
        Returns the posterior predictive distribution.

        Returns
        -------
        evidence.distributions.PosteriorPredictive
            The predictive distribution.
        """
        raise NotImplementedError

    @property
    def mean_distribution(self):
        """
        Computes the mean of this distribution and returns the likelihood distribution parametrized
        with this mean.

        Returns
        -------
        torch.distributions.ExponentialFamily
            The distribution that is defined by :meth:`__likelihood__`.
        """
        raise NotImplementedError

#--------------------------------------------------------------------------------------------------

class PosteriorPredictive(D.Distribution):
    """
    A posterior predictive distribution, typically obtained from a :class:`ConjugatePrior`.
    """

    def pvalue(self, x):
        """
        Computes the p-value of the given data for use in a two-sided statistical test.

        Parameters
        ----------
        x: torch.Tensor [N]
            The targets for which to compute the p-values.

        Returns
        -------
        torch.Tensor [N]
            The p-values.
        """
        cdf = self.cdf(x)
        return 2 * torch.min(cdf, 1 - cdf)
