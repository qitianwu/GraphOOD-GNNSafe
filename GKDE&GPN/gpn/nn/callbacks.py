from typing import Dict, Any
import logging
from sacred import Experiment
import pyblaze.nn as xnn
from pyblaze.nn.callbacks import TrainingCallback
from gpn.utils import TrainingConfiguration
from .early_stopping import AverageEarlyStopping, MultipleMetricEarlyStopping


class SacredTracker(TrainingCallback):
    """wrapper for pyblaze-based TrainingCallback to monitor metrics within the sacred framework
    """

    def __init__(self, experiment: Experiment):
        """
        Initializes a new tracker for the given sacred experiment.
        Parameters
        ----------
        experiment: sacred.Experiment
            The experiment to log for.
        """
        self.experiment = experiment

    def after_epoch(self, metrics: Dict[str, Any]) -> None:
        for m_name, m_value in metrics.items():
            self.experiment.log_scalar(m_name, m_value)



def get_callbacks_from_config(train_cfg: TrainingConfiguration) -> list:
    """[summary]

    Args:
        train_cfg (TrainingConfiguration): specified training configuration

    Returns:
        list: list of callback objects as specified in the training configuration
    """

    callbacks = []

    if train_cfg.stopping_mode is not None:
        if train_cfg.stopping_patience > 0:
            mode = train_cfg.stopping_mode

            if mode == 'default':
                early_stopping = xnn.callbacks.EarlyStopping

            elif mode == 'average':
                early_stopping = AverageEarlyStopping

            elif mode == 'multiple':
                early_stopping = MultipleMetricEarlyStopping

            else:
                warn_str = f'stopping_mode {mode} not implemented, falling back to default!'
                logging.warning(warn_str)
                early_stopping = xnn.callbacks.EarlyStopping

            patience = train_cfg.stopping_patience
            callbacks.append(early_stopping(
                patience=patience,
                restore_best=train_cfg.stopping_restore_best,
                metric=train_cfg.stopping_metric,
                minimize=train_cfg.stopping_minimize))

    return callbacks
