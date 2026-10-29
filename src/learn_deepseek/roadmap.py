"""
Pedagogical Roadmap: 7-Stage DeepSeek V3/R1 Architecture Masterclass.

Tracks curriculum progress across Multi-Head Latent Attention (MLA), DeepSeekMoE,
Multi-Token Prediction (MTP), DeepSeek-R1 GRPO Reasoning, DualPipe, and FP8 Serving.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class StageStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass
class Topic:
    id: str
    title: str
    summary: str
    completed: bool = False


@dataclass
class Stage:
    stage_number: int
    title: str
    description: str
    topics: List[Topic]
    status: StageStatus = StageStatus.NOT_STARTED

    @property
    def progress_percentage(self) -> float:
        if not self.topics:
            return 0.0
        completed = sum(1 for t in self.topics if t.completed)
        return (completed / len(self.topics)) * 100.0


class DeepSeekRoadmap:
    """Manages progress and milestones for the 7-Stage DeepSeek curriculum."""

    def __init__(self) -> None:
        self.stages: Dict[int, Stage] = {
            1: Stage(
                stage_number=1,
                title="Architecture, Model Topology & The DeepSeek Revolution",
                description="DeepSeek-V3 671B (37B active) MoE topology, training compute cost efficiency ($6M vs $100M+), and high-level architectural pillars.",
                topics=[
                    Topic("1.1", "DeepSeek-V3 Total vs Active Parameters", "Understanding 671B total weights with 37B activated per token (1 shared + 8 routed experts).", True),
                    Topic("1.2", "Ultra-Low Training Compute Budget", "How algorithmic efficiency achieved frontier performance for $5.9M on 2,048 H800 GPUs.", True),
                    Topic("1.3", "Open Weights & Model Distillation", "Distilling R1 reasoning patterns into Qwen and Llama architectures.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            2: Stage(
                stage_number=2,
                title="Multi-Head Latent Attention (MLA) Deep-Dive",
                description="Low-rank joint KV compression into 512-dim latent space, decoupled RoPE, and 93.3% KV cache memory reduction over MHA.",
                topics=[
                    Topic("2.1", "Low-Rank KV Compression Mechanics", "Projecting hidden states into latent space c_KV and decompressing at runtime.", True),
                    Topic("2.2", "Decoupled Rotary Position Embedding (RoPE)", "Separating content keys from positional keys to enable cache-invariant RoPE.", True),
                    Topic("2.3", "KV Cache Memory Sizing (MLA vs GQA vs MHA)", "Mathematical proof of 93.3% memory reduction unlocking massive batch concurrency.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            3: Stage(
                stage_number=3,
                title="DeepSeekMoE & Multi-Token Prediction (MTP)",
                description="Shared experts + routed experts, fine-grained expert division (256 experts), load balancing loss, and MTP speculative decoding.",
                topics=[
                    Topic("3.1", "Shared + Fine-Grained Routed Experts", "Allocating 1 dedicated shared expert + 8 dynamically routed experts out of 256.", True),
                    Topic("3.2", "Auxiliary-Loss-Free Load Balancing", "Dynamic routing bias adjustment preventing expert collapse without gradient penalties.", True),
                    Topic("3.3", "Multi-Token Prediction (MTP)", "Predicting k consecutive tokens in parallel, achieving 1.8x speculative speedups.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            4: Stage(
                stage_number=4,
                title="DeepSeek-R1: Pure Reinforcement Learning & Emergent Reasoning",
                description="DeepSeek-R1-Zero pure RL without SFT, Group Relative Policy Optimization (GRPO), rule-based accuracy/format rewards, and the 'Aha moment'.",
                topics=[
                    Topic("4.1", "DeepSeek-R1-Zero & Emergent Thinking", "How pure RL produces self-reflection, backtracking, and exploration without human demos.", True),
                    Topic("4.2", "Group Relative Policy Optimization (GRPO)", "Eliminating the critic neural network by computing baseline advantages across output groups.", True),
                    Topic("4.3", "Rule-Based Reward Functions", "Compiler execution, math verification, and <think>...</think> XML formatting rewards.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            5: Stage(
                stage_number=5,
                title="DualPipe Distributed Parallelism & FP8 Mixed Precision",
                description="Overlapping forward/backward computation with cross-node communication, chunked pipeline scheduling, and E4M3/E5M2 FP8 math.",
                topics=[
                    Topic("5.1", "DualPipe Bidirectional Overlapping", "Full computation-communication overlap hiding cross-node All-to-All MoE latency.", True),
                    Topic("5.2", "Tile-Wise & Block-Wise FP8 Quantization", "Using 1x128 tile-wise scaling factors to preserve gradient dynamic range.", True),
                    Topic("5.3", "InfiniBand Cross-Node Communication", "Optimizing collective communication primitives for dense MoE dispatch.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            6: Stage(
                stage_number=6,
                title="Self-Hosting, Quantization & Production Inference Serving",
                description="Serving 671B models with vLLM / SGLang, AWQ / GGUF quantization tiers, multi-node deployment, and memory budgets.",
                topics=[
                    Topic("6.1", "Inference Engines (vLLM & SGLang)", "Optimized MLA and MoE CUDA kernels for ultra-fast time-to-first-token (TTFT).", True),
                    Topic("6.2", "Hardware Cluster Sizing", "Estimating 8x H100 / A100 vs 16x nodes for FP8 / INT4 quantized deployment.", True),
                    Topic("6.3", "Distilled Model Deployment", "Serving DeepSeek-R1-Distill-Llama-8B / 70B on single-node workstations.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
            7: Stage(
                stage_number=7,
                title="Staff-Level Interview Prep, Cheatsheet & Appendix",
                description="DeepSeek systems architectural comparison matrix, failure runbooks, and staff interview technical questions.",
                topics=[
                    Topic("7.1", "Architectural Comparisons", "DeepSeek-V3 vs Llama 3 vs GPT-4o vs Claude 3.5 Sonnet across all system dimensions.", True),
                    Topic("7.2", "Production Incident Runbooks", "Handling MoE routing hotspot skews, InfiniBand packet drops, and FP8 underflows.", True),
                    Topic("7.3", "Master Interview Q&A", "Deep-dive answers to 40+ staff-level MLA, MoE, GRPO, and distributed systems questions.", True),
                ],
                status=StageStatus.COMPLETED,
            ),
        }

    def get_stage(self, stage_number: int) -> Optional[Stage]:
        return self.stages.get(stage_number)

    def total_progress(self) -> float:
        total_topics = sum(len(s.topics) for s in self.stages.values())
        if total_topics == 0:
            return 0.0
        completed = sum(sum(1 for t in s.topics if t.completed) for s in self.stages.values())
        return (completed / total_topics) * 100.0

    def get_summary(self) -> Dict[str, object]:
        return {
            "total_stages": len(self.stages),
            "completed_stages": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "overall_progress_pct": round(self.total_progress(), 2),
            "stages": [
                {
                    "stage": s.stage_number,
                    "title": s.title,
                    "status": s.status.value,
                    "progress_pct": round(s.progress_percentage, 1),
                    "topics_count": len(s.topics),
                }
                for s in self.stages.values()
            ],
        }
