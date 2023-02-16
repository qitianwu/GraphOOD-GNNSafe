from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from sacred import Experiment
from gpn.layers import CertaintyDiffusion
from gpn.utils import apply_mask
from gpn.utils import Prediction
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from gpn.utils import Storage, ModelNotFoundError


class Model(nn.Module):
    """base model which provides functionality to load and store models, compute losses, specify matching optimizers, and much more"""

    def __init__(self, params: ModelConfiguration):
        super().__init__()
        self._expects_training = True
        self._is_warming_up = False
        self._is_finetuning = False

        if params is not None:
            self.params = params.clone()

        self.storage = None
        self.storage_params = None
        self.model_file_path = None
        self.cached_y = None

    def forward(self, data: Data, *_, **__) -> Prediction:
        x = self.forward_impl(data)
        log_soft = F.log_softmax(x, dim=-1)
        soft = torch.exp(log_soft)
        max_soft, hard = soft.max(dim=-1)

        # cache soft prediction for SGCN, for which a
        # model might act as teacher (e.g. GAT/GCN)
        self.cached_y = soft

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            logits=x,
            # confidence of prediction
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=None,
            prediction_confidence_structure=None,
            # confidence of sample
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=None,
            sample_confidence_features=None,
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def expects_training(self) -> bool:
        return self._expects_training

    def is_warming_up(self) -> bool:
        return self._is_warming_up

    def is_finetuning(self) -> bool:
        return self._is_finetuning

    def set_expects_training(self, flag: bool) -> None:
        self._expects_training = flag

    def set_warming_up(self, flag: bool) -> None:
        self._is_warming_up = flag

    def set_finetuning(self, flag: bool) -> None:
        self._is_finetuning = flag

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_impl(self, data: Data, *args, **kwargs):
        raise NotImplementedError

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        return self.CE_loss(prediction, data)

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def fintetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def CE_loss(self, prediction: Prediction, data: Data, reduction='mean') -> Dict[str, torch.Tensor]:
        y_hat = prediction.log_soft
        y_hat, y = apply_mask(data, y_hat, split='train')

        return {
            'CE': F.nll_loss(y_hat, y, reduction=reduction)
        }

    def save_to_file(self, model_path: str) -> None:
        save_dict = {
            'model_state_dict': self.state_dict(),
            'cached_y': self.cached_y
        }
        torch.save(save_dict, model_path)

    def load_from_file(self, model_path: str) -> None:
        if not torch.cuda.is_available():
            c = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            c = torch.load(model_path)
        self.load_state_dict(c['model_state_dict'])
        self.cached_y = c['cached_y']

    def get_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        raise NotImplementedError

    def create_storage(self, run_cfg: RunConfiguration, data_cfg: DataConfiguration,
                       model_cfg: ModelConfiguration, train_cfg: TrainingConfiguration,
                       ex: Experiment = None):

        if run_cfg.job == 'train' or (run_cfg.job == 'evaluate' and run_cfg.eval_experiment_name is None):
            run_cfg.set_values(eval_experiment_name=run_cfg.experiment_name)

        storage = Storage(run_cfg.experiment_directory,
                          experiment_name=run_cfg.eval_experiment_name,
                          experiment=ex)

        storage_params = {**model_cfg.to_dict(ignore=model_cfg.default_ignore()),
                          **data_cfg.to_dict(), **train_cfg.to_dict()}

        # ignore ood parameters for matching in an evasion setting
        if run_cfg.job == 'evaluate' and data_cfg.ood_flag and data_cfg.ood_setting == 'evasion':
            storage_params = {k: v for k, v in storage_params.items() if not k.startswith('ood_')}

        self.storage = storage
        self.storage_params = storage_params

    def load_from_storage(self) -> None:
        if self.storage is None:
            raise ModelNotFoundError('Error on loading model, storage does not exist!')

        model_file_path = self.storage.retrieve_model_file_path(
            self.storage_params['model_name'], self.storage_params, init_no=self.params.init_no
        )

        self.load_from_file(model_file_path)

    def save_to_storage(self) -> None:
        if self.storage is None:
            raise ModelNotFoundError('Error on storing model, storage does not exist!')

        model_file_path = self.storage.create_model_file_path(
            self.storage_params['model_name'], self.storage_params, init_no=self.params.init_no
        )

        self.save_to_file(model_file_path)
