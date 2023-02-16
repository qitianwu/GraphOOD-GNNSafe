from typing import Optional
from torch import Tensor
import pyblaze.nn.functional as X
from gpn.utils import Prediction
from .metrics import brier_score, confidence
from .metrics import expected_calibration_error, maximum_calibration_error
from .metrics import ood_detection
from .metrics import ood_detection_features
from .metrics import ood_detection_structure
from .metrics import ood_detection_neighborhood
from .metrics import average_confidence, average_entropy
from .loss import cross_entropy


def get_metrics(metrics):
    """get the functions implementing metrics from a list of strings naming those

    Args:
        metrics (list): list of metric names

    Returns:
        list: list of functions implementing those metrics
    """

    if metrics is None:
        metrics = {}

    metrics_dict = {}

    for m in metrics:
        _m = get_metric(m)
        metrics_dict[_m[0]] = _m[1]

    return metrics_dict


def get_metric(metric: str):
    """return the function that implemented the passed metric

    Args:
        metric (str): name of the metric

    Raises:
        NotImplementedError: raised if passed metric is not supported

    Returns:
        lambda: function that implemented the desired metric
    """

    metric = metric.lower()

    # basic metrics
    if metric == 'accuracy':
        return metric, lambda y_hat, y: _metric_wrapper(
            X.metrics.accuracy, y_hat, y, key='hard')

    if metric == 'f1_score':
        return metric, lambda y_hat, y: _metric_wrapper(
            X.metrics.f1_score, y_hat, y, key='hard')

    if metric == 'brier_score':
        return metric, lambda y_hat, y: _metric_wrapper(
            brier_score, y_hat, y, key='soft')

    if metric == 'ece':
        return 'ECE', lambda y_hat, y: _metric_wrapper(
            expected_calibration_error, y_hat, y, key=None)

    if metric == 'mce':
        return 'MCE', lambda y_hat, y: _metric_wrapper(
            maximum_calibration_error, y_hat, y, key=None)

    if metric == 'ce':
        return 'CE', lambda y_hat, y: _metric_wrapper(
            cross_entropy, y_hat, y, key='soft')

    if metric == 'confidence_aleatoric_auroc':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='AUROC', uncertainty_type='aleatoric')

    if metric == 'confidence_aleatoric_apr':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='APR', uncertainty_type='aleatoric')

    if metric == 'confidence_epistemic_auroc':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='AUROC', uncertainty_type='epistemic')

    if metric == 'confidence_epistemic_apr':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='APR', uncertainty_type='epistemic')

    if metric == 'confidence_structure_auroc':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='AUROC', uncertainty_type='structure')

    if metric == 'confidence_structure_apr':
        return metric, lambda y_hat, y: _metric_wrapper(
            confidence, y_hat, y, key=None, score_type='APR', uncertainty_type='structure')

    if metric == 'avg_prediction_confidence_aleatoric':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='prediction', uncertainty_type='aleatoric')

    if metric == 'avg_prediction_confidence_epistemic':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='prediction', uncertainty_type='epistemic')

    if metric == 'avg_sample_confidence_aleatoric':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='sample', uncertainty_type='aleatoric')

    if metric == 'avg_sample_confidence_epistemic':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='sample', uncertainty_type='epistemic')

    if metric == 'avg_sample_confidence_features':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='sample', uncertainty_type='features')

    if metric == 'avg_sample_confidence_neighborhood':
        return metric, lambda y_hat, y: _metric_wrapper(
            average_confidence, y_hat, y, key=None, confidence_type='sample', uncertainty_type='neighborhood')

    if metric == 'average_entropy':
        return 'average_entropy', lambda y_hat, y: _metric_wrapper(average_entropy, y_hat, y, key=None)

    # metrics for ood detection
    if metric == 'ood_detection_aleatoric_auroc':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection, y_hat, y, y_hat_ood, y_ood, key=None, score_type='AUROC', uncertainty_type='aleatoric')

    if metric == 'ood_detection_aleatoric_apr':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection, y_hat, y, y_hat_ood, y_ood, key=None, score_type='APR', uncertainty_type='aleatoric')

    if metric == 'ood_detection_epistemic_auroc':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection, y_hat, y, y_hat_ood, y_ood, key=None, score_type='AUROC', uncertainty_type='epistemic')

    if metric == 'ood_detection_epistemic_apr':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection, y_hat, y, y_hat_ood, y_ood, key=None, score_type='APR', uncertainty_type='epistemic')

    if metric == 'ood_detection_features_auroc':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_features, y_hat, y, y_hat_ood, y_ood, key=None, score_type='AUROC')

    if metric == 'ood_detection_features_apr':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_features, y_hat, y, y_hat_ood, y_ood, key=None, score_type='APR')

    if metric == 'ood_detection_neighborhood_auroc':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_neighborhood, y_hat, y, y_hat_ood, y_ood, key=None, score_type='AUROC')

    if metric == 'ood_detection_neighborhood_apr':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_neighborhood, y_hat, y, y_hat_ood, y_ood, key=None, score_type='APR')

    if metric == 'ood_detection_structure_auroc':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_structure, y_hat, y, y_hat_ood, y_ood, key=None, score_type='AUROC')

    if metric == 'ood_detection_structure_apr':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            ood_detection_structure, y_hat, y, y_hat_ood, y_ood, key=None, score_type='APR')

    # metrics on ood nodes
    if metric == 'ood_accuracy':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            X.metrics.accuracy, y_hat, y, y_hat_ood, y_ood, key='hard', setting='ood')

    if metric == 'ood_avg_prediction_confidence_aleatoric':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='prediction', uncertainty_type='aleatoric')

    if metric == 'ood_avg_prediction_confidence_epistemic':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='prediction', uncertainty_type='epistemic')

    if metric == 'ood_avg_sample_confidence_aleatoric':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='sample', uncertainty_type='aleatoric')

    if metric == 'ood_avg_sample_confidence_epistemic':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='sample', uncertainty_type='epistemic')

    if metric == 'ood_avg_sample_confidence_features':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='sample', uncertainty_type='features')

    if metric == 'ood_avg_sample_confidence_neighborhood':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood',
            confidence_type='sample', uncertainty_type='neighborhood')

    if metric == 'ood_average_entropy':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_entropy, y_hat, y, y_hat_ood, y_ood, key=None, setting='ood')

    # metrics on id nodes
    if metric == 'id_accuracy':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            X.metrics.accuracy, y_hat, y, y_hat_ood, y_ood, key='hard', setting='id')

    if metric == 'id_avg_prediction_confidence_aleatoric':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='prediction', uncertainty_type='aleatoric')

    if metric == 'id_avg_prediction_confidence_epistemic':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='prediction', uncertainty_type='epistemic')

    if metric == 'id_avg_sample_confidence_aleatoric':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='sample', uncertainty_type='aleatoric')

    if metric == 'id_avg_sample_confidence_epistemic':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='sample', uncertainty_type='epistemic')

    if metric == 'id_avg_sample_confidence_features':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='sample', uncertainty_type='features')

    if metric == 'id_avg_sample_confidence_neighborhood':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_confidence, y_hat, y, y_hat_ood, y_ood, key=None, setting='id',
            confidence_type='sample', uncertainty_type='neighborhood')

    if metric == 'id_average_entropy':
        return metric, lambda y_hat, y, y_hat_ood, y_ood: _ood_metric_wrapper(
            average_entropy, y_hat, y, y_hat_ood, y_ood, key=None, setting='id')

    raise NotImplementedError(f'{metric} currently not implemented!')


def _metric_wrapper(metric: callable, y_hat: Prediction, y: Tensor, key: Optional[str] = None, **kwargs):
    """convenience function for easily computing metrics from model predictions"""
    
    if key is not None:
        y_hat = getattr(y_hat, key)
        return metric(y_hat, y, **kwargs)

    return metric(y_hat, y, **kwargs)


def _ood_metric_wrapper(metric: callable, y_hat_id: Prediction, y_id: Tensor, 
                        y_hat_ood: Prediction, y_ood: Tensor, key: Optional[str] = None, setting: str = 'combined', **kwargs):
    """convenience function for easily computing OOD metrics from model predictions"""
    
    assert setting in ('combined', 'id', 'ood')

    if key is not None:
        y_hat_id = getattr(y_hat_id, key)
        y_hat_ood = getattr(y_hat_ood, key)

    if setting == 'combined':
        return metric(y_hat_id, y_id, y_hat_ood, y_ood, **kwargs)

    if setting == 'id':
        return metric(y_hat_id, y_id, **kwargs)

    if setting == 'ood':
        return metric(y_hat_ood, y_ood, **kwargs)

    raise AssertionError
