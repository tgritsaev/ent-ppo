from dataclasses import dataclass, field
from gflownet.utils.misc import StrictDataClass


@dataclass
class GraphTransformerConfig(StrictDataClass):
    num_heads: int = 2
    ln_type: str = "pre"
    num_mlp_layers: int = 0
    concat_heads: bool = True


@dataclass
class ModelConfig(StrictDataClass):
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    unif_init: bool = True
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
