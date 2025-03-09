
from .PredFormer import PredFormer

method_maps = {
    'predformer': PredFormer,
}

__all__ = [
    'method_maps', 'PredFormer'
]