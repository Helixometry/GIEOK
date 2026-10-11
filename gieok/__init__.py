"""GIEOK: Paralinguistic Audio-Language Modeling for Zero-Shot Cross-Lingual Alzheimer's Detection"""

__version__ = "1.0.0"
__author__ = "Girish, Mohd Mujtaba Akhtar, Muskaan Singh, Juliana Gerard, Paula McClean, Kongfatt Wong-Lin"
__email__ = "m.singh@ulster.ac.uk"

from .model import GIEOK
from .modules import SCAF, TemporalAdapter, CueAdapter
from .data import load_manifest, create_splits

__all__ = [
    "GIEOK",
    "SCAF",
    "TemporalAdapter",
    "CueAdapter",
    "load_manifest",
    "create_splits",
]
