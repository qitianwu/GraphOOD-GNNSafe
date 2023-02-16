import copy
import numpy as np
from pyblaze.nn.callbacks import EarlyStopping, CallbackException


class AverageEarlyStopping(EarlyStopping):
    """early stopping based on metric averages"""

    def __init__(self, metric='val_loss', patience=10, restore_best=True, minimize=True):
        super().__init__(metric, patience, restore_best, minimize)
        self.metric_history = []

    def after_epoch(self, metrics):
        prev_epoch = self.epoch
        self.epoch += 1

        try:
            is_better = self._is_metric_better(metrics)
        except KeyError:
            if prev_epoch > 0:
                # In this case, we can ignore the key error and just skip -- the engine does not
                # perform evaluation on every iteration
                return

        if is_better:
            if self.restore_best:
                self.state_dict = copy.deepcopy(self.model.state_dict())
            self.counter = 0
        else:
            # metric is only better when window met, i.e. return immediately
            raise CallbackException(
                f"Early stopping after epoch {self.epoch} (patience {self.patience}).",
                verbose=True
            )
    
    # current metric is better if epoch < patience or better than average window
    def _is_metric_better(self, metrics):
        self._current_metric(metrics)
        if self.minimize:
            return self.epoch < self.patience or self.metric_history[-1] < np.mean(
                self.metric_history[-(self.patience + 1):-1])
        else:
            return self.epoch < self.patience or self.metric_history[-1] > np.mean(
                self.metric_history[-(self.patience + 1):-1])          

    # simply append current metric to history
    def _current_metric(self, metrics):
        metric = metrics[self.metric]
        self.metric_history.append(metric)


class MultipleMetricEarlyStopping(EarlyStopping):
    """early stopping considering multiple metrics"""

    def __init__(self, metric=('val_loss', 'val_acc'), patience=10,
                 restore_best=True, minimize=(True, False)):

        if isinstance(metric, str):
            metric = [metric]
        if isinstance(minimize, bool):
            minimize = [minimize]

        assert isinstance(metric, (list, tuple))
        assert isinstance(minimize, (list, tuple))
        assert len(metric) == len(minimize)
        self._is_better = [None] * len(metric)

        super().__init__(metric, patience, restore_best, minimize)
       
    def before_training(self, model, num_epochs):
        self.model = model
        if self.restore_best:
            self.state_dict = copy.deepcopy(model.state_dict())
        self.epoch = 0
        self.counter = 0
        self.best_metric = [float('inf') if m else -float('inf') for m in self.minimize]

    def after_epoch(self, metrics):
        prev_epoch = self.epoch
        self.epoch += 1
        current_metrics = self._current_metric(metrics)

        try:
            is_better = self._is_metric_better(current_metrics)
        except KeyError:
            if prev_epoch > 0:
                # In this case, we can ignore the key error and just skip -- the engine does not
                # perform evaluation on every iteration
                return

        if is_better:
            if self.restore_best:
                self.state_dict = copy.deepcopy(self.model.state_dict())
            self.counter = 0
            self._best_metric(current_metrics)
        else:
            self.counter += 1
            if self.counter == self.patience:
                raise CallbackException(
                    f"Early stopping after epoch {self.epoch} (patience {self.patience}).",
                    verbose=True)

    def _current_metric(self, metrics):
        return [metrics[m] for m in self.metric]
    
    def _is_metric_better(self, current_metrics):
        for i, m in enumerate(current_metrics):
            if self.minimize[i]:
                self._is_better[i] = m < self.best_metric[i]
            else:
                self._is_better[i] = m > self.best_metric[i]
        # better if at least one metric is better
        return any(self._is_better)

    def _best_metric(self, current_metrics):
        for i, m in enumerate(current_metrics):
            if self.minimize[i]:
                self.best_metric[i] = np.min((self.best_metric[i], m))
            else:
                self.best_metric[i] = np.max((self.best_metric[i], m))
