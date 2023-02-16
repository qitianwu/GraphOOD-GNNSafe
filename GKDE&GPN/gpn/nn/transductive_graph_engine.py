import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch

from pyblaze.nn.callbacks import CallbackException
from pyblaze.nn.callbacks import TrainingCallback, PredictionCallback, \
    ValueTrainingCallback
from pyblaze.nn.engine._history import History
from pyblaze.nn.engine import Engine
from pyblaze.nn.engine.base import _strip_metrics
from pyblaze.utils.torch import gpu_device, _recursive_apply

from gpn.utils import apply_mask


class TransductiveGraphEngine(Engine):
    def __init__(self, model, splits=('train', 'test', 'val', 'all')):
        super().__init__(model)
        self.splits = splits
        self.current_it = 0

    ##################################################################################

    def supports_multiple_gpus(self):
        return False

    def before_epoch(self, current, num_iterations):
        self.current_it = 0

    def after_batch(self, *_):
        if self.current_it is not None:
            self.current_it += 1

    def after_epoch(self, metrics):
        self.current_it = 0

    ################################################################################
    ### TRAINING
    ################################################################################
    def train(self, train_data, val_data=None, epochs=20, eval_every=None,
              eval_train=False, eval_val=True, callbacks=None, metrics=None, gpu='auto', **kwargs):

        if metrics is None:
            metrics = {}
        if callbacks is None:
            callbacks = []

        # 1) Setup
        try:
            batch_iterations = len(train_data)
            iterable_data = False
        except: # pylint: disable=bare-except
            batch_iterations = eval_every
            iterable_data = True

        exception = None
        if iterable_data and eval_every is not None:
            # Here, epochs are considered iterations
            epochs = epochs // eval_every

        # 1.1) Callbacks
        history = History()
        # Prepend the engine's callbacks to the passed callbacks
        callbacks = [history] + callbacks
        # Also, add the callbacks that are extracted from the keyword arguments
        callbacks += [v for _, v in kwargs.items() if isinstance(v, TrainingCallback)]
        # Then, we can extract the callbacks for training and prediction
        train_callbacks = [c for c in callbacks if isinstance(c, TrainingCallback)]
        prediction_callbacks = [c for c in callbacks if isinstance(c, PredictionCallback)]
        self._exec_callbacks(train_callbacks, 'before_training', self.model, epochs)

        # 1.2) Metrics
        val_metrics = metrics

        # 1.3) Data loading
        if iterable_data:
            train_iterator = iter(train_data)

        # 1.4) GPU support
        gpu = self._gpu_descriptor(gpu)
        self._setup_device(gpu)
        self.model.to(self.device)

        # 1.5) Valid kwargs
        train_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('eval_')}
        dynamic_train_kwargs = {
            k: v for k, v in train_kwargs.items() if isinstance(v, ValueTrainingCallback)
        }
        eval_kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith('eval_')}
        dynamic_eval_kwargs = {
            k: v for k, v in eval_kwargs.items() if isinstance(v, ValueTrainingCallback)
        }

        # 2) Train for number of epochs
        for current_epoch in range(epochs):
            # 2.1) Prepare
            try:
                self._exec_callbacks(
                    train_callbacks, 'before_epoch', current_epoch, batch_iterations
                )
            except CallbackException as e:
                exception = e
                break

            # 2.2) Train
            self.model.train()

            batch_losses = []
            if not iterable_data:
                train_iterator = iter(train_data)

            for i in range(batch_iterations):
                train_kwargs = {
                    **train_kwargs,
                    **{k: v.read() for k, v in dynamic_train_kwargs.items()}
                }
                item = next(train_iterator)
                item = item.to(self.device)
                loss = self.train_batch(item, **train_kwargs)
                batch_losses.append(loss)
                self._exec_callbacks(train_callbacks, 'after_batch', _strip_metrics(loss))

            # 2.3) Validate
            epoch_metrics = self.collate_losses(batch_losses)
            eval_kwargs = {
                **eval_kwargs,
                **{k: v.read() for k, v in dynamic_eval_kwargs.items()}
            }
            do_val = eval_every is None or iterable_data or \
                current_epoch % eval_every == 0 or current_epoch == epochs - 1

            if val_data is not None and do_val:
                eval_metrics = self.evaluate(
                    val_data, metrics=val_metrics,
                    callbacks=prediction_callbacks, gpu=None, **eval_kwargs
                )

                eval_metrics_val = eval_metrics['val']
                epoch_metrics = {**epoch_metrics, **{f'val_{k}': v for k, v in eval_metrics_val.items()}}

                if eval_train:
                    eval_metrics_train = eval_metrics['train']
                    epoch_metrics = {**epoch_metrics, **{f'train_{k}': v for k, v in eval_metrics_train.items()}}

            # 2.4) Finish epoch
            try:
                self._exec_callbacks(train_callbacks, 'after_epoch', epoch_metrics)
            except CallbackException as e:
                exception = e
                break

        # 3) Finish training
        # 3.1) If GPU used
        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        # 3.2) Finish callbacks
        self._exec_callbacks(train_callbacks, 'after_training')
        if exception is not None:
            if isinstance(exception, CallbackException):
                exception.print()
            else:
                print(exception)

        return history

    ################################################################################
    ### EVALUATION
    ################################################################################
    def evaluate(self, data, metrics=None, callbacks=None, gpu='auto', **kwargs):

        if metrics is None:
            metrics = {}

        evals = self._get_evals(data, gpu=gpu, callbacks=callbacks, **kwargs)
        return self._aggregate_metrics(evals, metrics)

    def evaluate_target_and_ood(self, data, data_ood, metrics=None, metrics_ood=None, 
                                callbacks=None, gpu='auto', target_as_id=True, **kwargs):

        if metrics is None:
            metrics = {}
        if metrics_ood is None:
            metrics_ood = {}

        evals = self._get_evals(data, callbacks=callbacks, gpu=gpu, **kwargs)
        evals_ood = self._get_evals(
            data_ood, callbacks=callbacks,
            gpu=gpu, split_prefix='ood', **kwargs)

        if target_as_id:
            # target represents ID values, e.g. when evaluating isolated perturbations
            # or leave-out-class experiments
            evals_id = evals
        else:
            # for usual evasion setting: id values correspond to non-perturbed nodes
            # while ood nodes correspond to perturbed nodes
            # target corresponds to evaluation of all nodes without this distinction
            evals_id = self._get_evals(
                data_ood, callbacks=callbacks,
                gpu=gpu, split_prefix='id', **kwargs)

        results = self._aggregate_metrics(evals, metrics)
        ood_results = self._aggregate_metrics_ood(evals_id, evals_ood, metrics_ood)

        for s in self.splits:
            results[s] = {**results[s], **ood_results[s]}

        return results

    ################################################################################
    ### PREDICTIONS
    ################################################################################
    def predict(self, data, callbacks=None, gpu='auto', parallel=True, **kwargs):

        if callbacks is None:
            callbacks = []

        # 1) Set gpu if all is specified
        gpu = self._gpu_descriptor(gpu)

        # 2) Setup data loading
        num_iterations = len(data)

        self._exec_callbacks(callbacks, 'before_predictions', self.model, num_iterations)

        # 3) Now perform predictions

        # sequential computation
        device = gpu_device(gpu[0] if isinstance(gpu, list) else gpu)
        self.model.to(device)

        predictions = []

        iterator = iter(data)
        for _ in range(num_iterations):
            item = next(iterator)
            item = item.to(self.device)

            with torch.no_grad():
                out = self.predict_batch(item, **kwargs)
            out = out.to('cpu')

            predictions.append(out)
            self._exec_callbacks(callbacks, 'after_batch', None)

        self._exec_callbacks(callbacks, 'after_predictions')

        return self.collate_predictions(predictions)


    ################################################################################
    ### BATCH PROCESSING
    ################################################################################

    def eval_batch(self, data, split_prefix=None, **kwargs):
        assert split_prefix in (None, 'ood', 'id')

        y_hat = self.predict_batch(data)
        evals = {}
        for s in self.splits:
            if s == 'all':
                if split_prefix in ('ood', 'id'):
                    _y_hat, _y = apply_mask(data, y_hat, split_prefix)
                else:
                    _y_hat, _y = y_hat, data.y
            else:
                if split_prefix is None:
                    _s = s
                else:
                    _s = f'{split_prefix}_{s}'

                _y_hat, _y = apply_mask(data, y_hat, _s)

            evals[s] = _y_hat, _y

        return evals

    def train_batch(self, data, optimizer, loss=None, **kwargs):
        # train full model
        optimizer.zero_grad()
        y_hat = self.predict_batch(data)

        if loss is None:
            loss_train = self.model.loss(y_hat, data)

        else:
            loss_train = loss(y_hat, data)

        loss = 0.0
        loss_dict = {}
        for l_key, l_val in loss_train.items():
            loss += l_val
            loss_dict[l_key] = l_val.detach().cpu().item()

        loss.backward()
        optimizer.step()

        return loss_dict

    def predict_batch(self, data, **kwargs):
        return self.model(data)


    ################################################################################
    ### COLLATION FUNCTIONS
    ################################################################################

    def collate_evals(self, evals):
        collated = {}
        for s in self.splits:
            collated[s] = (
                self._collate([e[s][0] for e in evals]),
                self._collate([e[s][1] for e in evals]),
            )

        return collated

    def collate_predictions(self, predictions):
        return predictions

    ################################################################################
    ### UTILITY FUNCTIONS
    ################################################################################
    def _aggregate_metrics(self, evals, metrics):
        metric_results = {}
        for s in self.splits:
            y_hat, y = evals[s]
            metric_results[s] = {
                metric_key: self._process_metric(metric(y_hat, y))
                for metric_key, metric in metrics.items()
            }

        return metric_results

    def _aggregate_metrics_ood(self, evals, evals_ood, metrics_ood):
        metric_results = {}
        for s in self.splits:
            y_hat, y = evals[s]
            y_hat_ood, y_ood = evals_ood[s]

            metric_results[s] = {
                metric_key: self._process_metric(metric(y_hat, y, y_hat_ood, y_ood))
                for metric_key, metric in metrics_ood.items()
            }

        return metric_results

    def _get_evals(self, data, callbacks=None, gpu='auto', **kwargs):

        if callbacks is None:
            callbacks = []

        # setup
        num_predictions = len(data)
        self._exec_callbacks(callbacks, 'before_predictions', self.model, num_predictions)

        # ensure GPU
        if gpu is not None:
            gpu = self._gpu_descriptor(gpu)
            self._setup_device(gpu)
            self.model.to(self.device)

        # run inference
        self.model.eval()

        evals = []
        iterator = iter(data)
        for _ in range(num_predictions):
            item = next(iterator)
            item = item.to(self.device)

            with torch.no_grad():
                eval_out = self.eval_batch(item, **kwargs)

            evals.append(self.to_device('cpu', eval_out))
            self._exec_callbacks(callbacks, 'after_batch', None)

        self._exec_callbacks(callbacks, 'after_predictions')

        evals = self.collate_evals(evals)

        if gpu is not None:
            self.model.to('cpu', non_blocking=True)
            self.device = None

        return evals

    def _collate(self, items):
        ref = items[0]
        if isinstance(ref, dict):
            return {key: torch.cat([v[key] for v in items]) for key in ref.keys()}
        # recursive call of _collate for a more generic functionality
        if isinstance(ref, (list, tuple)):
            return tuple(self._collate([v[i] for v in items]) for i in range(len(ref)))
        # if object implements _collate itself
        if callable(getattr(ref, 'collate', None)):
            return ref.collate(items[1:])

        return torch.cat(items)

    def to_device(self, device, item):
        def _to_device(x):
            return x.to(device, non_blocking=True)

        return _recursive_apply('to', _to_device, item)
