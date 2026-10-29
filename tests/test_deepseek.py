"""
Comprehensive Unit Tests for DeepSeek V3 & R1 Architecture Engine.

Tests:
- Multi-Head Latent Attention (MLA) low-rank KV compression & memory sizing.
- DeepSeekMoE shared + routed expert selection and weight normalization.
- Multi-Token Prediction (MTP) speculative generation speedup.
- DeepSeek-R1 Group Relative Policy Optimization (GRPO) rule-based rewards.
- DualPipe pipeline parallelism bubble reduction.
- Tile-wise FP8 mixed precision quantization.
- Pedagogical roadmap stage completion.
"""

import unittest

from src.learn_deepseek.architecture import (
    DeepSeekMoE,
    DualPipeScheduler,
    FP8Quantizer,
    GRPORewardEvaluator,
    MultiHeadLatentAttention,
    MultiTokenPredictor,
)
from src.learn_deepseek.roadmap import DeepSeekRoadmap


class TestDeepSeekEngine(unittest.TestCase):

    def test_mla_kv_compression(self) -> None:
        mla = MultiHeadLatentAttention(d_model=7168, n_heads=128, d_head=128, d_c=512, d_r=64)
        sizing = mla.calculate_kv_cache_bytes_per_token()

        # MHA: 2 * 128 * 128 * 2 = 65,536 bytes
        self.assertEqual(sizing["mha_bytes_per_token"], 65536)
        # MLA: (512 + 64) * 2 = 1,152 bytes
        self.assertEqual(sizing["mla_bytes_per_token"], 1152)

        # MLA must achieve >90% savings over MHA
        self.assertTrue(sizing["mha_savings_pct"] > 90.0)
        self.assertTrue(sizing["mha_compression_ratio"] > 50.0)

    def test_mla_compress_kv(self) -> None:
        mla = MultiHeadLatentAttention(d_c=512, d_r=64)
        hidden = [0.5] * 100
        c_kv, k_r = mla.compress_kv(hidden)
        self.assertEqual(len(c_kv), 512)
        self.assertEqual(len(k_r), 64)

    def test_moe_routing(self) -> None:
        moe = DeepSeekMoE(n_shared=1, n_routed=256, top_k=8)
        token_vec = [1.0, 2.0, 3.0, 4.0]
        routed = moe.route_token(token_vec)

        self.assertEqual(routed["shared_expert_id"], 0)
        self.assertEqual(len(routed["selected_routed_experts"]), 8)
        self.assertEqual(len(routed["routed_expert_weights"]), 8)
        self.assertEqual(routed["total_active_experts"], 9)
        # Softmax weights sum to ~1.0
        self.assertAlmostEqual(sum(routed["routed_expert_weights"]), 1.0, places=2)

    def test_multi_token_prediction(self) -> None:
        mtp = MultiTokenPredictor(num_future_tokens=2)
        res = mtp.predict_speculative("async", ["def", "fetch_data"])
        self.assertEqual(res["primary_token"], "async")
        self.assertEqual(len(res["accepted_stream"]), 3)
        self.assertEqual(res["speculative_speedup"], 3.0)

    def test_grpo_reward_evaluator(self) -> None:
        prompt = "Compute 10 * 10."
        expected = "100"
        completions = [
            "<think>10 times 10 is 100.</think> The answer is 100.",
            "<think>10 times 10 is 20.</think> The answer is 20.",
            "The answer is 100.",  # Missing thinking tags
        ]

        evaluated = GRPORewardEvaluator.evaluate_group(prompt, completions, expected)
        self.assertEqual(len(evaluated), 3)

        # Sample 0 has both format and accuracy reward
        self.assertEqual(evaluated[0]["accuracy_reward"], 1.0)
        self.assertEqual(evaluated[0]["format_reward"], 1.0)
        self.assertEqual(evaluated[0]["total_reward"], 1.0)

        # Sample 1 has format reward but 0 accuracy
        self.assertEqual(evaluated[1]["accuracy_reward"], 0.0)
        self.assertEqual(evaluated[1]["format_reward"], 1.0)

        # Sample 2 has accuracy reward but 0 format
        self.assertEqual(evaluated[2]["accuracy_reward"], 1.0)
        self.assertEqual(evaluated[2]["format_reward"], 0.0)

        # Sample 0 should have the highest advantage
        self.assertTrue(evaluated[0]["grpo_advantage"] > evaluated[1]["grpo_advantage"])
        self.assertTrue(evaluated[0]["grpo_advantage"] > evaluated[2]["grpo_advantage"])

    def test_dualpipe_scheduler(self) -> None:
        bubble = DualPipeScheduler.calculate_pipeline_bubble(8, 32)
        self.assertAlmostEqual(bubble["bubble_reduction_pct"], 50.0, places=1)
        self.assertTrue(bubble["dualpipe_bubble_ratio"] < bubble["standard_1f1b_bubble_ratio"])

    def test_fp8_quantizer(self) -> None:
        vals = [float(i) for i in range(128)]
        tile = FP8Quantizer.quantize_tile(vals, tile_size=128)
        self.assertEqual(tile["tile_size"], 128)
        self.assertEqual(tile["compression_ratio"], 4.0)
        self.assertTrue(tile["tile_scale"] > 0)

    def test_roadmap_progress(self) -> None:
        roadmap = DeepSeekRoadmap()
        self.assertEqual(len(roadmap.stages), 7)
        self.assertEqual(roadmap.total_progress(), 100.0)
        summary = roadmap.get_summary()
        self.assertEqual(summary["total_stages"], 7)
        self.assertEqual(summary["completed_stages"], 7)
        self.assertEqual(summary["overall_progress_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
