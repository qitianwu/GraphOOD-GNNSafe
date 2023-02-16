from typing import Tuple
import torch
from torch import Tensor
import torch.distributions as D
import numpy as np
from sklearn import metrics
from gpn.utils import Prediction



def expected_calibration_error(y_hat: Prediction, y: Tensor, n_bins: int = 10) -> Tensor:
    """calculates the expected calibration error

    Args:
        y_hat (Prediction): model predictions
        y (Tensor): ground-truth labels
        n_bins (int, optional): number of bins used in the ECE calculation. Defaults to 10.

    Returns:
        Tensor: ECE
    """

    if (y_hat.soft is None) or (y_hat.hard is None):
        return torch.as_tensor(float('nan'))

    batch_size = y_hat.soft.size(0)
    if batch_size == 0:
        return torch.as_tensor(float('nan'))

    acc_binned, conf_binned, bin_cardinalities = bin_predictions(y_hat, y, n_bins)
    ece = torch.abs(acc_binned - conf_binned) * bin_cardinalities
    ece = ece.sum() * 1 / batch_size
    return ece.cpu().detach()


def maximum_calibration_error(y_hat: Prediction, y: Tensor, n_bins: int = 10) -> Tensor:
    """calculates the maximum calibration error

    Args:
        y_hat (Prediction): model predictions
        y (Tensor): ground-truth labels
        n_bins (int, optional): number of bins used in the MCE calculation. Defaults to 10.

    Returns:
        Tensor: MCE
    """

    if (y_hat.soft is None) or (y_hat.hard is None):
        return torch.as_tensor(float('nan'))

    batch_size = y_hat.soft.size(0)
    if batch_size == 0:
        return torch.as_tensor(float('nan'))

    acc_binned, conf_binned, _ = bin_predictions(y_hat, y, n_bins)
    mce = torch.abs(acc_binned - conf_binned)
    mce = torch.max(mce)

    return mce.cpu().detach()


def brier_score(y_hat: Tensor, y: Tensor) -> Tensor:
    """calculates the Brier score

    Args:
        y_hat (Tensor): predicted class probilities
        y (Tensor): ground-truth labels

    Returns:
        Tensor: Brier Score
    """
    batch_size = y_hat.size(0)
    if batch_size == 0:
        return torch.as_tensor(float('nan'))
    prob = y_hat.clone()
    indices = torch.arange(batch_size)
    prob[indices, y] -= 1

    return prob.norm(dim=-1, p=2).mean().detach().cpu()


def confidence(y_hat: Prediction, y: Tensor, score_type: str = 'AUROC', uncertainty_type: str = 'aleatoric') -> Tensor:
    """calculates AUROC/APR scores based on different confidence values (relevant for misclassification experiments)

    Args:
        y_hat (Prediction): model predictions
        y (Tensor): ground-truth labels
        score_type (str, optional): score type (either AUROC or APR). Defaults to 'AUROC'.
        uncertainty_type (str, optional): uncertainty scores used in calculation. Defaults to 'aleatoric'.

    Returns:
        Tensor: confidence scores
    """

    corrects = (y.squeeze() == y_hat.hard).cpu().detach().int().numpy()

    key = f'prediction_confidence_{uncertainty_type}'

    if getattr(y_hat, key) is not None:
        scores = getattr(y_hat, key).cpu().detach().numpy()

        if len(scores) == 0:
            return torch.as_tensor(float('nan'))

        return _area_under_the_curve(score_type, corrects, scores)

    return torch.as_tensor(float('nan'))


def average_confidence(y_hat: Prediction, _, confidence_type: str = 'prediction', 
                       uncertainty_type: str = 'aleatoric') -> Tensor:
    """calculates the average confidence scores involved in the prediction (either for prediction or uncertainty in general)

    Args:
        y_hat (Prediction): models prediction
        _ (Any): placeholder for pipeline compatibility
        confidence_type (str, optional): desired confidence type. Defaults to 'prediction'.
        uncertainty_type (str, optional): desired uncertainty type. Defaults to 'aleatoric'.

    Returns:
        Tensor: average confidence
    """

    key = f'{confidence_type}_confidence_{uncertainty_type}'

    if getattr(y_hat, key) is not None:
        return getattr(y_hat, key).mean()

    return torch.as_tensor(float('nan'))


def average_entropy(y_hat: Prediction, _) -> Tensor:
    """calculates the average entropy over all nodes in the prediction

    Args:
        y_hat (Prediction): models prediction
        _ (Any): placeholder for pipeline compatibility

    Returns:
        Tensor: average entropy
    """
    entropy = D.Categorical(y_hat.soft).entropy().mean()
    return entropy


def ood_detection(y_hat: Prediction, _, y_hat_ood: Prediction, __,
                  score_type: str = 'AUROC', uncertainty_type: str = 'aleatoric') -> Tensor:
    """convenience function which computes the OOD APR/AUROC scores from model predictions on ID and OOD data based on estimates of 'aleatoric' or 'epistemic' uncertainty

    Args:
        y_hat (Prediction): model predictions for ID data
        _ (Any): placeholder for pipeline compatibility
        y_hat_ood (Prediction): model predictions for OOD data
        __ (Any): placeholder for pipeline compatibility
        score_type (str, optional): 'APR' or 'AUROC'. Defaults to 'AUROC'.
        uncertainty_Type (str, optional): 'aleatoric' or 'epistemic'. Defaults to 'aleatoric'

    Returns:
        Tensor: APR/AUROC scores
    """

    # for compatibility with metrics-API: targets y also passed
    key = f'sample_confidence_{uncertainty_type}'
    return _ood_detection(y_hat, y_hat_ood, key=key, score_type=score_type)


def ood_detection_features(y_hat: Prediction, _, y_hat_ood: Prediction, __,
                           score_type: str = 'AUROC') -> Tensor:
    """convenience function which computes the OOD APR/AUROC scores from model predictions on ID and OOD data based on estimates of 'feature' uncertainty

    Args:
        y_hat (Prediction): model predictions for ID data
        _ (Any): placeholder for pipeline compatibility
        y_hat_ood (Prediction): model predictions for OOD data
        __ (Any): placeholder for pipeline compatibility
        score_type (str, optional): 'APR' or 'AUROC'. Defaults to 'AUROC'.

    Returns:
        Tensor: APR/AUROC scores
    """

    return _ood_detection(y_hat, y_hat_ood, key='sample_confidence_features', score_type=score_type)


def ood_detection_neighborhood(y_hat: Prediction, _, y_hat_ood: Prediction, __,
                               score_type: str = 'AUROC') -> Tensor:
    """convenience function which computes the OOD APR/AUROC scores from model predictions on ID and OOD data based on estimates of 'neighborhood' uncertainty

    Args:
        y_hat (Prediction): model predictions for ID data
        _ (Any): placeholder for pipeline compatibility
        y_hat_ood (Prediction): model predictions for OOD data
        __ (Any): placeholder for pipeline compatibility
        score_type (str, optional): 'APR' or 'AUROC'. Defaults to 'AUROC'.

    Returns:
        Tensor: APR/AUROC scores
    """

    return _ood_detection(y_hat, y_hat_ood, key='sample_confidence_neighborhood', score_type=score_type)


def ood_detection_structure(y_hat: Prediction, _, y_hat_ood: Prediction, __,
                            score_type: str = 'AUROC') -> Tensor:
    """convenience function which computes the OOD APR/AUROC scores from model predictions on ID and OOD data based on estimates of 'structural' uncertainty

    Args:
        y_hat (Prediction): model predictions for ID data
        _ (Any): placeholder for pipeline compatibility
        y_hat_ood (Prediction): model predictions for OOD data
        __ (Any): placeholder for pipeline compatibility
        score_type (str, optional): 'APR' or 'AUROC'. Defaults to 'AUROC'.

    Returns:
        Tensor: APR/AUROC scores
    """

    return _ood_detection(y_hat, y_hat_ood, key='sample_confidence_structure', score_type=score_type)


def _ood_detection(y_hat: Prediction, y_hat_ood: Prediction, key: str, score_type: str) -> Tensor:
    """interntal convenience function to compute APR/AUROC scores for OOD detection based on predictions on ID and OOD data

    Args:
        y_hat (Prediction): predictions on ID data
        y_hat_ood (Prediction): predictions on OOD data
        key (str): uncertainty scores to use for calculation of APR/AUROC scores, e.g. sample_confidence_structure
        score_type (str): 'APR' or 'AUROC'

    Returns:
        Tensor: APR/AUROC scores
    """
    y_hat = getattr(y_hat, key)
    y_hat_ood = getattr(y_hat_ood, key)

    if (y_hat is not None) and (y_hat_ood is not None):
        scores = y_hat.cpu().detach().numpy()
        ood_scores = y_hat_ood.cpu().detach().numpy()

    else:
        return torch.as_tensor(float('nan'))

    if (len(scores) == 0) or (len(ood_scores) == 0):
        return torch.as_tensor(float('nan'))

    n_id = scores.shape[0]
    n_ood = ood_scores.shape[0]

    # for OOD detection: use reversed problem, i.e. actually consider
    # OOD samples as target 1 and ID data with target 0
    corrects = np.concatenate([np.zeros(n_id), np.ones(n_ood)], axis=0)
    scores = np.concatenate([scores, ood_scores], axis=0)
    # invert scores
    scores = -scores

    return _area_under_the_curve(score_type, corrects, scores)


def _area_under_the_curve(score_type: str, corrects: np.array, scores: np.array) -> Tensor:
    """calculates the area-under-the-curve score (either PR or ROC)

    Args:
        score_type (str): desired score type (either APR or AUROC)
        corrects (np.array): binary array indicating correct predictions
        scores (np.array): array of prediction scores

    Raises:
        AssertionError: raised if score other than APR or AUROC passed

    Returns:
        Tensor: area-under-the-curve scores
    """
    # avoid INF or NAN values
    scores = np.nan_to_num(scores)

    if score_type == 'AUROC':
        fpr, tpr, _ = metrics.roc_curve(corrects, scores)
        return torch.as_tensor(metrics.auc(fpr, tpr))

    if score_type == 'APR':
        prec, rec, _ = metrics.precision_recall_curve(corrects, scores)
        return torch.as_tensor(metrics.auc(rec, prec))

    raise AssertionError


def bin_predictions(y_hat: Prediction, y: Tensor, n_bins: int = 10) -> Tuple[Tensor, Tensor, Tensor]:
    """bins predictions based on predicted class probilities

    Args:
        y_hat (Prediction): predicted class probabilities
        y (Tensor): ground-truth labels
        n_bins (int, optional): number of bins used in the calculation. Defaults to 10.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: tuple of binned accuracy values, confidence values and cardinalities of each bin
    """
    y_hat, y_hat_label = y_hat.soft, y_hat.hard
    y_hat = y_hat.max(-1)[0]
    corrects = (y_hat_label == y.squeeze())

    acc_binned = torch.zeros((n_bins, ), device=y_hat.device)
    conf_binned = torch.zeros((n_bins, ), device=y_hat.device)
    bin_cardinalities = torch.zeros((n_bins, ), device=y_hat.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    lower_bin_boundary = bin_boundaries[:-1]
    upper_bin_boundary = bin_boundaries[1:]

    for b in range(n_bins):
        in_bin = (y_hat <= upper_bin_boundary[b]) & (y_hat > lower_bin_boundary[b])
        bin_cardinality = in_bin.sum()
        bin_cardinalities[b] = bin_cardinality

        if bin_cardinality > 0:
            acc_binned[b] = corrects[in_bin].float().mean()
            conf_binned[b] = y_hat[in_bin].mean()

    return acc_binned, conf_binned, bin_cardinalities
