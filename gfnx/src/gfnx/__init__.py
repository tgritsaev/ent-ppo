from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import networks

from . import metrics, spaces, utils
from .base import (
    TAction,
    TBackwardAction,
    TDone,
    TEnvironment,
    TEnvParams,
    TEnvState,
    TLogReward,
    TObs,
    TReward,
    TRewardModule,
    TRewardParams,
)
from .environment import (
    HypergridEnvironment,
    HypergridEnvParams,
    HypergridEnvState,
    QM9SmallEnvironment,
    QM9SmallEnvParams,
    QM9SmallEnvState,
    TFBind8Environment,
    TFBind8EnvParams,
    TFBind8EnvState,
)
from .reward import (
    EasyHypergridRewardModule,
    GeneralHypergridRewardModule,
    HardHypergridRewardModule,
    QM9SmallRewardModule,
    TFBind8RewardModule,
)
from .visualize import Visualizer

__all__ = [
    "metrics",
    "networks",
    "spaces",
    "utils",
    "EasyHypergridRewardModule",
    "GeneralHypergridRewardModule",
    "HardHypergridRewardModule",
    "HypergridEnvironment",
    "HypergridEnvParams",
    "HypergridEnvState",
    "TAction",
    "TFBind8Environment",
    "TFBind8EnvParams",
    "TFBind8EnvState",
    "TFBind8RewardModule",
    "QM9SmallEnvironment",
    "QM9SmallEnvParams",
    "QM9SmallEnvState",
    "QM9SmallRewardModule",
    "TBackwardAction",
    "TDone",
    "TEnvParams",
    "TEnvState",
    "TEnvironment",
    "TLogReward",
    "TObs",
    "TReward",
    "TRewardModule",
    "TRewardParams",
    "Visualizer",
]

# Lazy import of networks since networks are based on Equinox
import importlib


def __getattr__(name):
    if name == "networks":
        return importlib.import_module(f"{__name__}.networks")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return __all__
