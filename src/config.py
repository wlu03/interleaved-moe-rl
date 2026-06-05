from dataclasses import dataclass


@dataclass
class InterleavedMoEConfig:
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1376  # dense FFN expansion (~2.7x hidden)
    num_experts: int = 4
    num_experts_per_tok: int = 2
    expert_intermediate_size: int = 1376
    moe_every_n_layers: int = 2  # MoE on layers 0, 2, 4, 6; dense on 1, 3, 5, 7
    vocab_size: int = 151936  # Qwen2.5-Math tokenizer
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
