from dataclasses import dataclass, field
from gflownet.utils.misc import StrictDataClass


@dataclass
class SEHTaskConfig(StrictDataClass):
    reduced_frag: bool = False
    large_test_mols_path: str = ""


@dataclass
class QM9TaskConfig(StrictDataClass):
    h5_path: str = "qm9.h5"  # see src/gflownet/data/qm9.py
    model_path: str = "mxmnet_gap_model.pt"
    rdkit_conformer_timeout_seconds: int = 0


@dataclass
class TasksConfig(StrictDataClass):
    qm9: QM9TaskConfig = field(default_factory=QM9TaskConfig)
    seh: SEHTaskConfig = field(default_factory=SEHTaskConfig)
