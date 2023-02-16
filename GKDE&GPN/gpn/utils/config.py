from typing import Union, List, Tuple
import attr
from .object import HalfFrozenObject


@attr.s(frozen=True)
class RunConfiguration(HalfFrozenObject):
    """object specifying possible job configurations"""

    # experiment name (relevant for saving/loading trained models)
    experiment_name: str = attr.ib(default=None)

    # experiment-name for evaluation (relevant e.g. for evasion attacks)
    eval_experiment_name: str = attr.ib(default=None)

    # root directory to save/load models to/from
    experiment_directory: str = attr.ib(default=None)

    # evaluation mode
    #   default: e.g. ood evasion or re-evaluation
    #   dropout: i.e. DropoutEnsemble evaluation
    #   ensemble: i.e. Ensemble evaluation (using model-init 1-10!!!)
    eval_mode: str = attr.ib(default=None, validator=lambda i, a, v: v in (
        'default', 'dropout', 'ensemble', 'energy_scoring'))

    # flag whether to run a job as experiment ("train": training + evaluation)
    # or only in "evaluation" mode (e.g. re-evaluating model,
    # evlulating models on other datasets, or as dropout-models or ensembles)
    job: str = attr.ib(default=None, validator=lambda i, a, v: v in ('train', 'evaluate'))

    # save-flag (e.g. for not saving GridSearch experiments)
    save_model: bool = attr.ib(default=None)

    # gpu
    gpu: int = attr.ib(default=None, validator=lambda i, a, v: v in (0, False))

    # run multiple experiments at one
    num_inits: int = attr.ib(default=None)
    num_splits: int = attr.ib(default=None)

    # running experiment
    log: bool = attr.ib(default=True) # flag for logging training progress and metrics
    debug: bool = attr.ib(default=True) # flag for running code in a "DEBUG" mode
    ex_type: str = attr.ib(default='transductive', validator=lambda i, a, v: v in (
        'transductive', 'transductive_ood'))

    ood_loc: bool = attr.ib(default=True) # flag for running LOC in ood_experiment
    ood_loc_only: bool = attr.ib(default=False) # flag for only runninig LOC in ood_experiment

    ood_edge_perturbations: bool = attr.ib(default=True) # flag for running edge pert. exp. in ood_experiment
    ood_isolated_perturbations: bool = attr.ib(default=False) # flag for running isolated exp. in ood_experiment


@attr.s(frozen=True)
class DataConfiguration(HalfFrozenObject):
    """object specifying possible dataset configurations"""

    # sparseness
    to_sparse: bool = attr.ib(default=False)

    # ranomness
    split_no: int = attr.ib(default=None, validator=lambda i, a, v: v is not None and v > 0)

    # dataset parameters
    dataset: str = attr.ib(default=None)
    root: str = attr.ib(default=None)
    split: str = attr.ib(default=None, validator=lambda i, a, v: v in ('public', 'random'))
    # note that either the num-examples for the size values
    # must be specified, but not both at the same time!
    train_samples_per_class: Union[int, float] = attr.ib(default=None)
    val_samples_per_class: Union[int, float] = attr.ib(default=None)
    test_samples_per_class: Union[int, float] = attr.ib(default=None)
    train_size: float = attr.ib(default=None)
    val_size: float = attr.ib(default=None)
    test_size: float = attr.ib(default=None)

    # ood parameters
    ood_flag: bool = attr.ib(default=False)
    ood_setting: str = attr.ib(default=None, validator=lambda i, a, v: v in ('evasion', 'poisoning', None))
    ood_type: str = attr.ib(default=None, validator=lambda i, a, v: v in (
        None, 'perturb_features', 'leave_out_classes',
        'leave_out_classes_evasion', 'random_attack_dice', 'random_attack_targeted', 'random_edge_perturbations'))
    ood_dataset_type: str = attr.ib(None, validator=lambda i, a, v: v in ('budget', 'isolated', None))
    # type of feature perturabtion, e.g. bernoulli_0.5
    ood_perturbation_type: str = attr.ib(default=None)
    ood_budget_per_graph: float = attr.ib(default=None)
    ood_budget_per_node: float = attr.ib(default=None)
    ood_noise_scale: float = attr.ib(default=None)
    ood_num_left_out_classes: int = attr.ib(default=None)
    ood_frac_left_out_classes: float = attr.ib(default=None)
    ood_left_out_classes: List[int] = attr.ib(default=None)
    ood_leave_out_last_classes: bool = attr.ib(default=None)


@attr.s(frozen=True)
class ModelConfiguration(HalfFrozenObject):
    """object specifying possible model configurations"""

    # model name
    model_name: str = attr.ib(default=None, validator=lambda i, a, v: v is not None and len(v) > 0)
    # randomness
    seed: int = attr.ib(default=None, validator=lambda i, a, v: v is not None and v > 0)
    init_no: int = attr.ib(default=None, validator=lambda i, a, v: v is not None and v > 0)

    # default parameters
    num_classes: int = attr.ib(default=None)
    dim_features: int = attr.ib(default=None)
    dim_hidden: Union[int, List[int]] = attr.ib(default=None)
    dropout_prob: float = attr.ib(default=None)
    dropout_prob_adj: float = attr.ib(default=0.0)
    # mainly relevant for ogbn-arxiv
    batch_norm: bool = attr.ib(default=None)

    # for constrained linear layers
    k_lipschitz: float = attr.ib(default=None)

    # for deeper networks
    num_layers: int = attr.ib(default=None)

    # GAT
    heads_conv1: int = attr.ib(default=None)
    heads_conv2: int = attr.ib(default=None)
    negative_slope: float = attr.ib(default=None)
    coefficient_dropout_prob: float = attr.ib(default=None)

    # diffusion
    K: int = attr.ib(default=None)
    alpha_teleport: float = attr.ib(default=None)
    add_self_loops: bool = attr.ib(default=None)

    # PostNet / NormalizingFlows
    radial_layers: int = attr.ib(default=None)
    ft_radial_layers: int = attr.ib(default=None)
    maf_layers: int = attr.ib(default=None)
    ft_maf_layers: int = attr.ib(default=None)
    gaussian_layers: int = attr.ib(default=None)
    ft_gaussian_layers: int = attr.ib(default=None)
    dim_latent: int = attr.ib(default=None)
    alpha_evidence_scale: Union[int, str] = attr.ib(default=None)
    entropy_reg: float = attr.ib(default=None)
    factor_flow_lr: float = attr.ib(default=None)
    flow_weight_decay: float = attr.ib(default=None)
    share_flow: bool = attr.ib(default=None)
    use_batched_flow: bool = attr.ib(default=None) 
    pre_train_mode: str = attr.ib(default=None, validator=lambda i, a, v: v in ('encoder', 'flow', 'none', None))
    likelihood_type: str = attr.ib(
        default=None,
        validator=lambda i, a, v: v in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', 'none', None))
    gpn_loss_type: str = attr.ib(
        default=None
    )

    # Natural PostNets
    weight_evidence_transformation: str = attr.ib(default=None)
    weight_evidence_scale: float = attr.ib(default=None)
    latent_space_aggregation: str = attr.ib(default=None)
    loss_nll_weight: float = attr.ib(default=None)
    use_flow_mixture: bool = attr.ib(default=None)
    node_normalization: str = attr.ib(default=None)
    approximate_reg: bool = attr.ib(default=None)
    neighborhood_evidence: str = attr.ib(default=None)
    loss_reduction: str = attr.ib(default=None, validator=lambda i, a, v: v in (None, 'sum', 'mean'))
    loss_nll_weight_with_classes: bool = attr.ib(default=None)

    # RGCN
    gamma: float = attr.ib(default=None)
    beta_kl: float = attr.ib(default=None)
    beta_reg: float = attr.ib(default=None)

    # BayesianGCN
    bayesian_samples: int = attr.ib(default=None)
    pi: float = attr.ib(default=None)
    sigma_1: float = attr.ib(default=None)
    sigma_2: float = attr.ib(default=None)

    # DUN
    beta_dun: float = attr.ib(default=None)
    depth_in_message_passing: bool = attr.ib(default=None)

    # SGCN
    teacher_training: bool = attr.ib(default=None)
    teacher_params: dict = attr.ib(default=None)
    use_bayesian_dropout: bool = attr.ib(default=None)
    use_kernel: bool = attr.ib(default=None)
    lambda_1: float = attr.ib(default=None)
    sample_method: str = attr.ib(default=None, validator=lambda i, a, v: v in (None, 'log_evidence', 'alpha', 'none'))
    epochs: int = attr.ib(default=None)

    # dropout / ensemble
    num_samples_dropout: int = attr.ib(default=None)
    ensemble_min_init_no: int = attr.ib(default=None)
    ensemble_max_init_no: int = attr.ib(default=None)

    # scoring
    temperature: float = attr.ib(default=None)

    def default_ignore(self) -> List[str]:
        """define default attributes to ignore when loading/storing models
        """

        ignore = [
            'temperature',
            'ensemble_max_init_no',
            'ensemble_min_init_no',
            'num_samples_dropout',
            'init_no'
        ]

        for i in ignore:
            assert hasattr(self, i)

        return ignore


@attr.s(frozen=True)
class TrainingConfiguration(HalfFrozenObject):
    """object specifying possible training configurations"""

    lr: float = attr.ib(default=None)
    weight_decay: float = attr.ib(default=None)
    epochs: int = attr.ib(default=None)
    warmup_epochs: int = attr.ib(default=None)
    finetune_epochs: int = attr.ib(default=None)
    stopping_mode: str = attr.ib(default=None, validator=lambda i, a, v: v in (None, 'default', 'average', 'multiple'))
    stopping_patience: int = attr.ib(default=None)
    stopping_restore_best: bool = attr.ib(default=None)
    stopping_metric: str = attr.ib(default=None)
    stopping_minimize: bool = attr.ib(default=None)


def configs_from_dict(d: dict) -> Tuple[RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration]:
    """utility function converting a dictionary (e.g. coming from a .yaml file) into the corresponding configuration objects

    Args:
        d (dict): dictionary containing all relevant configuration parameters

    Returns:
        Tuple[RunConfiguration, DataConfiguration, ModelConfiguration, TrainingConfiguration]: tuple of corresponding objects for run, data, model, and training configuration
    """
    run = RunConfiguration(**d['run'])
    data = DataConfiguration(**d['data'])
    model = ModelConfiguration(**d['model'])
    training = TrainingConfiguration(**d['training'])

    return run, data, model, training
