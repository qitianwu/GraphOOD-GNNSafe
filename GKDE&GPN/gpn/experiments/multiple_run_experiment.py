from typing import Dict, Any
import copy
import numpy as np
from sacred import Experiment
from gpn.utils import RunConfiguration, DataConfiguration
from gpn.utils import ModelConfiguration, TrainingConfiguration
from .transductive_experiment import TransductiveExperiment


class MultipleRunExperiment:
    """wrapper for experiment which runs a experiment over for all init_no and split_no and aggregates the results"""

    def __init__(
            self,
            run_cfg: RunConfiguration,
            data_cfg: DataConfiguration,
            model_cfg: ModelConfiguration,
            training_cfg: TrainingConfiguration,
            ex: Experiment = None):

        self.run_cfg = run_cfg
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = training_cfg
        self.ex = ex

        if self.run_cfg.eval_mode == 'ensemble' or self.model_cfg.model_name in ('GDK', 'DiffusionRho', 'MaternGGP', 'GGP'):
            self.init_nos = [1]

        else:
            self.init_nos = [self.model_cfg.init_no] if self.run_cfg.num_inits is None else \
                range(1, self.run_cfg.num_inits + 1)

        self.split_nos = [self.data_cfg.split_no] if self.run_cfg.num_splits is None else \
            range(1, self.run_cfg.num_splits + 1)

        # disable logging when evaluating multiple inits and splits
        if len(self.split_nos) + len(self.init_nos) > 2:
            self.run_cfg.set_values(log=False)

    def run(self):
        run_results = []

        for split_no in self.split_nos:
            for init_no in self.init_nos:
                self.data_cfg.set_values(split_no=split_no)
                self.model_cfg.set_values(init_no=init_no)

                if self.run_cfg.ex_type == 'transductive':
                    results = self.run_transductive_experiment()

                else:
                    raise ValueError

                run_results.append(results)

        result_keys = run_results[0].keys()
        result_values = {k: [v[k] for v in run_results] for k in result_keys}
        result_means = {k: float(np.array(v).mean()) for k, v in result_values.items()}

        return_results = None
        # if only one configuration: behave as default experiment
        if len(run_results) == 1:
            return_results = result_means

        else:
            return_results = {
                **{f'{k}': v for k, v in result_means.items()},
                **{f'{k}_val': v for k, v in result_values.items()}
            }

        return return_results

    def run_transductive_experiment(self) -> Dict[str, Any]:
        experiment = TransductiveExperiment(
            self.run_cfg.clone(), self.data_cfg.clone(),
            self.model_cfg.clone(), self.train_cfg.clone(), ex=self.ex)
        results = experiment.run()

        return results