from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from gflownet.utils.misc import StrictDataClass


class Backward(IntEnum):
    """
    See algo.trajectory_balance.TrajectoryBalance for details.
    The A variant of `Maxent` and `GSQL` equire the environment to provide $n$.
    This is true for sEH but not QM9.
    """

    Uniform = 1
    Free = 2
    Maxent = 3
    MaxentA = 4
    GSQL = 5
    GSQLA = 6


class NLoss(IntEnum):
    """See algo.trajectory_balance.TrajectoryBalance for details."""

    none = 0
    Transition = 1
    SubTB1 = 2
    TermTB1 = 3
    StartTB1 = 4
    TB = 5


class TBVariant(IntEnum):
    """See algo.trajectory_balance.TrajectoryBalance for details."""

    TB = 0
    SubTB1 = 1
    DB = 2


class LossFN(IntEnum):
    """
    The loss function to use.

    - GHL:  Kaan Gokcesu, Hakan Gokcesu
    https://arxiv.org/pdf/2108.12627.pdf,
    Note: This can be used as a differentiable version of HUB.
    """

    MSE = 0
    MAE = 1
    HUB = 2
    GHL = 3


@dataclass
class TBConfig(StrictDataClass):
    """Trajectory Balance config.

    Attributes
    ----------
    bootstrap_own_reward : bool
        Whether to bootstrap the reward with the own reward. (deprecated)
    epsilon : Optional[float]
        The epsilon parameter in log-flow smoothing (see paper)
    reward_loss_multiplier : float
        The multiplier for the reward loss when bootstrapping the reward. (deprecated)
    variant : TBVariant
        The loss variant. See algo.trajectory_balance.TrajectoryBalance for details.
    do_correct_idempotent : bool
        Whether to correct for idempotent actions
    do_parameterize_p_b : bool
        Whether to parameterize the P_B distribution (otherwise it is uniform)
    do_predict_n : bool
        Whether to predict the number of paths in the graph
    do_length_normalize : bool
        Whether to normalize the loss by the length of the trajectory
    subtb_max_len : int
        The maximum length trajectories, used to cache subTB computation indices
    Z_learning_rate : float
        The learning rate for the logZ parameter (only relevant when do_subtb is False)
    Z_lr_decay : float
        The learning rate decay for the logZ parameter (only relevant when do_subtb is False)
    loss_fn: LossFN
        The loss function to use
    loss_fn_par: float
        The loss function parameter in case of Huber loss, it is the delta
    n_loss: NLoss
        The $n$ loss to use (defaults to NLoss.none i.e., do not learn $n$)
    n_loss_multiplier: float
        The multiplier for the $n$ loss
    backward_policy: Backward
        The backward policy to use
    gradient_steps: int
        Number of optimizer steps to take on each sampled training batch
    """

    bootstrap_own_reward: bool = False
    epsilon: Optional[float] = None
    reward_loss_multiplier: float = 1.0
    variant: TBVariant = TBVariant.TB
    do_correct_idempotent: bool = False
    do_parameterize_p_b: bool = False
    do_predict_n: bool = False
    do_sample_p_b: bool = False
    do_length_normalize: bool = False
    subtb_max_len: int = 128
    Z_learning_rate: float = 1e-4
    Z_lr_decay: float = 50_000
    cum_subtb: bool = True
    loss_fn: LossFN = LossFN.MSE
    loss_fn_par: float = 1.0
    n_loss: NLoss = NLoss.none
    n_loss_multiplier: float = 1.0
    backward_policy: Backward = Backward.Uniform
    gradient_steps: int = 1


@dataclass
class EntPPOConfig(StrictDataClass):
    clip_eps: float = 0.2
    gae_lambda: float = 0.7
    policy_updates: int = 4
    backward_updates: Optional[int] = None
    value_updates: int = 4
    value_num_splits: int = 8
    value_learning_rate: Optional[float] = None
    value_learning_rate_multiplier: float = 1 / 3
    value_loss_multiplier: float = 1.0
    kl_coeff: float = 1.0
    normalize_advantages: bool = False
    gamma: float = 1.0
    do_sample_p_b: bool = False


@dataclass
class AlgoConfig(StrictDataClass):
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    num_from_policy : int
        The number of on-policy samples for a training batch.
        If using a replay buffer, see `replay.num_from_replay` for the number of samples from the replay buffer, and
        `replay.num_new_samples` for the number of new samples to add to the replay buffer (e.g. `num_from_policy=0`,
        and `num_new_samples=N` inserts `N` new samples in the replay buffer at each step, but does not make that data
        part of the training batch).
    num_from_dataset : int
        The number of samples from the dataset for a training batch
    valid_num_from_policy : int
        The number of on-policy samples for a validation batch
    valid_num_from_dataset : int
        The number of samples from the dataset for a validation batch
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    train_random_action_prob : float
        The probability of taking a random action during training
    train_det_after: Optional[int]
        Do not take random actions after this number of steps
    valid_random_action_prob : float
        The probability of taking a random action during validation
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    num_from_policy: int = 64
    num_from_dataset: int = 0
    valid_num_from_policy: int = 64
    valid_num_from_dataset: int = 0
    valid_num_eval_trajectories: Optional[int] = None
    valid_num_eval_dataset_trajectories: Optional[int] = None
    valid_use_ema: bool = False
    valid_ema_tau: float = 0.95
    max_backward_steps: Optional[int] = None
    max_len: int = 128
    max_nodes: int = 128
    max_edges: int = 128
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.0
    train_random_action_prob_anneal_steps: Optional[int] = None
    train_det_after: Optional[int] = None
    valid_random_action_prob: float = 0.0
    sampling_tau: float = 0.0
    backward_approach: str = "uniform"
    backward_learning_rate_multiplier: float = 0.1
    backward_lr_decay: float = 20_000
    backward_ema_tau: float = 0.95
    use_backward_ema: bool = False
    tb: TBConfig = field(default_factory=TBConfig)
    ent_ppo: EntPPOConfig = field(default_factory=EntPPOConfig)
