"""Stage-1 HIA-to-Qwen alignment pre-training utilities."""

from .data import AlignmentBatch, AlignmentDataset, HIA_PHONE_TOKEN, HIA_UTT_TOKEN, HIA_WORD_TOKEN
from .hia_features import HiaFeatureExtractor, HiaFeatureOutput, pool_word_branch_by_word_id
from .modeling import HiaQwenAlignmentModel, ProjectorConfig

__all__ = [
    "AlignmentBatch",
    "AlignmentDataset",
    "HIA_PHONE_TOKEN",
    "HIA_UTT_TOKEN",
    "HIA_WORD_TOKEN",
    "HiaFeatureExtractor",
    "HiaFeatureOutput",
    "HiaQwenAlignmentModel",
    "ProjectorConfig",
    "pool_word_branch_by_word_id",
]
