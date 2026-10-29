"""
learn_deepseek: DeepSeek V3 & R1 Systems Architecture Engineering Engine.

Provides Multi-Head Latent Attention (MLA) KV compression, DeepSeekMoE shared/routed experts,
Multi-Token Prediction (MTP), Group Relative Policy Optimization (GRPO) rule-based rewards,
DualPipe distributed pipeline scheduling, and FP8 mixed precision arithmetic.
"""

from .architecture import (
    MultiHeadLatentAttention,
    DeepSeekMoE,
    MultiTokenPredictor,
    GRPORewardEvaluator,
    DualPipeScheduler,
    FP8Quantizer,
)
from .roadmap import DeepSeekRoadmap

__all__ = [
    "MultiHeadLatentAttention",
    "DeepSeekMoE",
    "MultiTokenPredictor",
    "GRPORewardEvaluator",
    "DualPipeScheduler",
    "FP8Quantizer",
    "DeepSeekRoadmap",
]
