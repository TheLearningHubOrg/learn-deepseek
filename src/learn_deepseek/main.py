"""
DeepSeek V3 & R1 Architecture Masterclass: CLI & Demonstrator.

Demonstrates:
1. 7-Stage Curriculum progress tracking.
2. Multi-Head Latent Attention (MLA) KV cache footprint reduction (93.3% over MHA).
3. DeepSeekMoE: 1 Shared Expert + Top-8 of 256 Fine-Grained Routed Experts.
4. Multi-Token Prediction (MTP) speculative generation speedup.
5. DeepSeek-R1 Group Relative Policy Optimization (GRPO) rule-based reward scoring.
6. DualPipe bidirectional pipeline parallelism bubble minimization.
"""

from src.learn_deepseek.architecture import (
    DeepSeekMoE,
    DualPipeScheduler,
    FP8Quantizer,
    GRPORewardEvaluator,
    MultiHeadLatentAttention,
    MultiTokenPredictor,
)
from src.learn_deepseek.roadmap import DeepSeekRoadmap


def run_roadmap_showcase() -> None:
    roadmap = DeepSeekRoadmap()
    summary = roadmap.get_summary()

    print("=" * 80)
    print("DEEPSEEK V3 & R1 SYSTEMS ARCHITECTURE MASTERCLASS: 7-STAGE CURRICULUM")
    print("=" * 80)
    print(f"Curriculum Completion: {summary['overall_progress_pct']}% | Stages: {summary['completed_stages']}/{summary['total_stages']} Completed\n")

    for s in summary["stages"]:
        status_symbol = "[x]" if s["status"] == "COMPLETED" else "[ ]"
        print(f"  {status_symbol} Stage {s['stage']}: {s['title']} ({s['topics_count']} topics, {s['progress_pct']}%)")
    print("=" * 80 + "\n")


def run_mla_demo() -> None:
    print("-" * 80)
    print("STAGE 2: MULTI-HEAD LATENT ATTENTION (MLA) KV CACHE SIZING")
    print("-" * 80)

    mla = MultiHeadLatentAttention(d_model=7168, n_heads=128, d_head=128, d_c=512, d_r=64)
    res = mla.calculate_kv_cache_bytes_per_token()

    print("Attention Mechanisms KV Memory per Token per Layer:")
    print(f"  1. Standard MHA (128 heads): {res['mha_bytes_per_token']:,} bytes/token (Baseline 100%)")
    print(f"  2. Grouped-Query (GQA, 8 grp): {res['gqa_bytes_per_token']:,} bytes/token ({res['gqa_savings_pct']}% savings over MHA)")
    print(f"  3. DeepSeek MLA (Compressed):  {res['mla_bytes_per_token']:,} bytes/token ({res['mha_compression_ratio']}x compression, {res['mha_savings_pct']}% savings over MHA!)")
    print("\n  Impact: Unlocks massive concurrency and multi-million token context windows in GPU HBM.\n")


def run_moe_demo() -> None:
    print("-" * 80)
    print("STAGE 3: DEEPSEEKMOE ARCHITECTURE (SHARED + 256 ROUTED EXPERTS)")
    print("-" * 80)

    moe = DeepSeekMoE(n_shared=1, n_routed=256, top_k=8)
    sample_token = [0.82, -0.45, 1.23, 0.12]
    routed = moe.route_token(sample_token)

    print(f"Token Routing Result (Total 257 Experts):")
    print(f"  Shared Expert:          Expert #{routed['shared_expert_id']} (Always active)")
    print(f"  Top-8 Routed Experts:   {routed['selected_routed_experts']}")
    print(f"  Expert Softmax Weights: {routed['routed_expert_weights']}")
    print(f"  Total Active Experts:   {routed['total_active_experts']} / 257\n")


def run_mtp_demo() -> None:
    print("-" * 80)
    print("STAGE 3: MULTI-TOKEN PREDICTION (MTP) SPECULATIVE DECODING")
    print("-" * 80)

    mtp = MultiTokenPredictor(num_future_tokens=1)
    res = mtp.predict_speculative(
        current_token="def",
        candidate_future=["quicksort"],
    )

    print(f"Primary Token:     '{res['primary_token']}'")
    print(f"Speculative Token: {res['speculative_tokens']}")
    print(f"Accepted Output:   {res['accepted_stream']}")
    print(f"Effective Speedup: {res['speculative_speedup']}x generation throughput\n")


def run_grpo_r1_demo() -> None:
    print("-" * 80)
    print("STAGE 4: DEEPSEEK-R1 GRPO RULE-BASED REWARD & REASONING")
    print("-" * 80)

    prompt = "Solve for x: 3x + 12 = 27."
    expected = "5"

    completions = [
        "<think>Let's subtract 12 from both sides: 3x = 15. Then divide by 3: x = 5.</think> The answer is 5.",
        "The answer is 5.",  # Lacks <think> tags
        "<think>Let's add 12 to 27 which gives 39. So x = 13.</think> The answer is 13.",  # Wrong answer
    ]

    evaluations = GRPORewardEvaluator.evaluate_group(prompt, completions, expected)

    print(f"Prompt: \"{prompt}\" (Expected Answer: '{expected}')\n")
    for idx, e in enumerate(evaluations, 1):
        print(f"  Sample #{idx}:")
        print(f"    Raw Completion:  \"{e['completion'][:60]}...\"")
        print(f"    Accuracy Reward: {e['accuracy_reward']} | Format Reward: {e['format_reward']} | Total: {e['total_reward']}")
        print(f"    GRPO Advantage:  {e['grpo_advantage']:+0.4f}")
    print()


def run_dualpipe_demo() -> None:
    print("-" * 80)
    print("STAGE 5: DUALPIPE BIDIRECTIONAL PIPELINE BUBBLE REDUCTION")
    print("-" * 80)

    stages = 8
    microbatches = 32
    bubble = DualPipeScheduler.calculate_pipeline_bubble(stages, microbatches)

    print(f"Cluster Configuration: {stages} Pipeline Stages, {microbatches} Microbatches")
    print(f"  Standard 1F1B Bubble Ratio: {bubble['standard_1f1b_bubble_ratio']:.4f}")
    print(f"  DualPipe Bubble Ratio:      {bubble['dualpipe_bubble_ratio']:.4f}")
    print(f"  Bubble Overhead Reduction:  {bubble['bubble_reduction_pct']}% faster training\n")


def main() -> None:
    run_roadmap_showcase()
    run_mla_demo()
    run_moe_demo()
    run_mtp_demo()
    run_grpo_r1_demo()
    run_dualpipe_demo()


if __name__ == "__main__":
    main()
