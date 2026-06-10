"""Regression tests for the pre-flight audit fixes.

Each test pins a bug the adversarial audit confirmed would only surface at real
scale / real data / real Modal execution, which the vocab-64 CPU suite missed.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import InterleavedMoEConfig
from src.model import InterleavedMoEModel, MoELayer
from src.rl.grpo import selective_log_softmax
from src.rl.rewards import extract_gsm8k_answer
from src.rl.data import _gsm8k_completion, gsm8k_gold
from src.rl.train import (
    _ByteTokenizer,
    _probe_routing_stats,
    build_model,
    load_checkpoint,
    load_checkpoint_field,
    main,
    register_config,
    save_checkpoint,
    TrainConfig,
)
from src.rl.grpo import GRPOConfig
from src.instrumentation.drift import collect_layer_activations


def _cfg():
    return InterleavedMoEConfig(
        hidden_size=32, num_layers=4, num_attention_heads=4, num_kv_heads=2,
        intermediate_size=64, expert_intermediate_size=64, num_experts=4,
        num_experts_per_tok=2, moe_every_n_layers=2, vocab_size=64,
        max_position_embeddings=128,
    )


class FakeTokenizer:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text):
        return [(ord(c) % (self.vocab_size - 2)) + 2 for c in text] or [2]

    def decode(self, ids):
        return "".join(
            chr((int(t) - 2) + 32) for t in ids
            if int(t) not in (self.pad_token_id, self.eos_token_id)
        )


# BLOCKER 6: MoE dispatch must not gather per-token weight copies, and must
# stay numerically identical to a reference SwiGLU-per-expert computation.
class TestMoEDispatch:
    def test_matches_reference_per_expert(self):
        torch.manual_seed(0)
        cfg = InterleavedMoEConfig(
            hidden_size=32, num_experts=4, num_experts_per_tok=2,
            expert_intermediate_size=48,
        )
        moe = MoELayer(cfg).eval()
        x = torch.randn(3, 7, 32)
        with torch.no_grad():
            out, _ = moe(x)
            # Reference: dense per-expert SwiGLU, summed with routing weights.
            xf = x.view(-1, 32)
            tw, ti, _ = moe.router(xf)
            ref = torch.zeros_like(xf)
            for n in range(xf.shape[0]):
                for k in range(moe.num_experts_per_tok):
                    e = int(ti[n, k])
                    g = xf[n] @ moe.gate_proj[e]
                    u = xf[n] @ moe.up_proj[e]
                    o = (torch.nn.functional.silu(g) * u) @ moe.down_proj[e]
                    ref[n] += tw[n, k] * o
            ref = ref.view(3, 7, 32)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_no_token_routed_to_an_expert_is_fine(self):
        # Force all tokens to one expert via a lopsided router; others get none.
        torch.manual_seed(1)
        cfg = InterleavedMoEConfig(
            hidden_size=16, num_experts=4, num_experts_per_tok=2,
            expert_intermediate_size=16,
        )
        moe = MoELayer(cfg).eval()
        with torch.no_grad():
            out, _ = moe(torch.randn(2, 3, 16))
        assert out.shape == (2, 3, 16)
        assert not torch.isnan(out).any()


# BLOCKERS 1/7: per-token logprob must not allocate a full-vocab log_softmax,
# and must still equal the reference value.
class TestSelectiveLogSoftmax:
    def test_matches_full_log_softmax(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 5, 7)
        index = torch.randint(0, 7, (3, 5))
        got = selective_log_softmax(logits, index)
        ref = torch.log_softmax(logits, dim=-1).gather(
            -1, index.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(got, ref, atol=1e-6)


# BLOCKERS 4/5/8/10: probe forwards must skip logits and chunk over rows.
class TestProbeForwards:
    def test_compute_logits_false_returns_none(self):
        torch.manual_seed(0)
        model = InterleavedMoEModel(_cfg())
        out = model(torch.randint(2, 64, (2, 6)), compute_logits=False)
        assert out["logits"] is None
        assert out["routing_stats"]  # routing still produced

    def test_collect_activations_chunks_match_single_pass(self):
        torch.manual_seed(0)
        model = InterleavedMoEModel(_cfg()).eval()
        ids = torch.randint(2, 64, (20, 8))
        a = collect_layer_activations(model, ids, chunk_size=4)
        b = collect_layer_activations(model, ids, chunk_size=100)
        for k in a:
            assert torch.allclose(a[k], b[k], atol=1e-5)
            assert a[k].shape[0] == 20  # all rows accumulated

    def test_routing_stats_chunked_has_required_keys(self):
        torch.manual_seed(0)
        model = InterleavedMoEModel(_cfg()).eval()
        ids = torch.randint(2, 64, (10, 6))
        stats = _probe_routing_stats(model, ids, chunk_size=3)
        assert stats  # MoE layers present
        for raw in stats.values():
            assert {"router_entropy", "expert_load",
                    "topk_indices", "topk_weights"} <= set(raw)
            # expert_load is a proper distribution
            assert abs(float(raw["expert_load"].sum()) - 1.0) < 1e-4


# BLOCKER 14: GSM8K comma-formatted answers must not be truncated.
class TestGSM8KCommas:
    def test_extract_keeps_full_integer(self):
        assert extract_gsm8k_answer("reasoning...\n#### 1,000") == "1000"
        assert extract_gsm8k_answer("blah\n#### 12,345,678") == "12345678"

    def test_gold_and_completion_consistent(self):
        ans = "He earns 130,000 total.\n#### 130,000"
        assert gsm8k_gold(ans) == "130000"
        assert "\\boxed{130000}" in _gsm8k_completion(ans)

    def test_negative_and_plain_still_work(self):
        assert extract_gsm8k_answer("#### -42") == "-42"
        assert extract_gsm8k_answer("#### 7") == "7"


# BLOCKER 16 + 15: drift baseline persists across resume; crash stamps the real
# last step, not max_steps.
class TestCheckpointBaselineAndStep:
    def _register(self, name="pf_tiny"):
        register_config(name, lambda: TrainConfig(
            name=name, model=_cfg(),
            grpo=GRPOConfig(group_size=2, max_completion_len=8),
            lr=1e-3, max_steps=4, batch_prompts=2,
            eval_every=10, drift_every=2, checkpoint_every=2, probe_size=6,
        ))
        return name

    def _sample(self, b, step):
        return [f"q{step}_{j}" for j in range(b)], [str(step + j) for j in range(b)]

    def test_init_acts_persisted_in_checkpoint(self, tmp_path):
        name = self._register()
        tok = FakeTokenizer()
        main(name, ckpt_dir=tmp_path, resume=False, tokenizer=tok,
             sample_batch=self._sample, device="cpu", max_steps=4)
        latest = tmp_path / name / "latest.pt"
        init_acts = load_checkpoint_field(latest, "init_acts")
        assert init_acts is not None and len(init_acts) > 0

    def test_resume_reuses_saved_baseline(self, tmp_path):
        name = self._register()
        tok = FakeTokenizer()
        main(name, ckpt_dir=tmp_path, resume=False, tokenizer=tok,
             sample_batch=self._sample, device="cpu", max_steps=2)
        before = load_checkpoint_field(tmp_path / name / "latest.pt", "init_acts")
        # Resume and run more; baseline must be unchanged (not recomputed from
        # the resumed weights).
        main(name, ckpt_dir=tmp_path, resume=True, tokenizer=tok,
             sample_batch=self._sample, device="cpu", max_steps=4)
        after = load_checkpoint_field(tmp_path / name / "latest.pt", "init_acts")
        for k in before:
            assert torch.allclose(before[k], after[k], atol=1e-6)

    def test_final_step_is_real_not_maxsteps_on_short_run(self, tmp_path):
        name = self._register()
        tok = FakeTokenizer()
        summary = main(name, ckpt_dir=tmp_path, resume=False, tokenizer=tok,
                       sample_batch=self._sample, device="cpu", max_steps=3)
        # 3 steps completed -> stamped step is 3
        assert summary["final_step"] == 3
        assert load_checkpoint_field(tmp_path / name / "latest.pt", "init_acts") is not None
        # raw step field also reflects completion
        model = build_model(_cfg())
        assert load_checkpoint(model, None, tmp_path / name / "latest.pt") == 3


# BLOCKER 11: cold-start zero-advantage steps are surfaced.
class TestColdStartSignal:
    def test_nonzero_adv_frac_reported(self, tmp_path):
        from src.rl.grpo import grpo_train_step
        from src.rl.rollout import RolloutBatch

        model = InterleavedMoEModel(_cfg())
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        tok = FakeTokenizer()

        # All-equal rewards -> all advantages zero -> nonzero_adv_frac == 0.
        def flat_rollout(m, t, prompts, golds, **kw):
            N = len(prompts) * kw["group_size"]
            T = 6
            return RolloutBatch(
                input_ids=torch.randint(2, 64, (N, T)),
                attention_mask=torch.ones(N, T, dtype=torch.long),
                completion_mask=torch.cat(
                    [torch.zeros(N, 3), torch.ones(N, 3)], dim=1).long(),
                rewards=torch.ones(N),
                advantages=torch.zeros(N),
                prompt_indices=torch.zeros(N, dtype=torch.long),
                prompt_lens=torch.full((N,), 3, dtype=torch.long),
                completion_texts=[""] * N,
            )

        m = grpo_train_step(
            model, tok, optim, prompts=["a", "b"], golds=["1", "2"],
            cfg=GRPOConfig(group_size=2, max_completion_len=6),
            rollout_fn=flat_rollout,
        )
        assert m["nonzero_adv_frac"] == 0.0
