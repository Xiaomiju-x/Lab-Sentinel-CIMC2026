#!/usr/bin/env python3
"""Auditable W8 nano-transformer definitions for Forge200 generative assets.

The module keeps configuration and parameter accounting independent from
PyTorch so the local contract tests can run without importing a training
stack.  ``build_model`` receives the already imported torch/nn modules on the
GPU worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VOCAB_SIZE = 2048
CONTEXT_TOKENS = 192
MAX_GENERATION_TOKENS = 24


@dataclass(frozen=True)
class NanoLMConfig:
    family: str
    vocab_size: int
    context_tokens: int
    max_generation_tokens: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    parameter_min: int
    parameter_max: int
    quantization: str = "W8"
    architecture: str = "CAUSAL_PRENORM_TRANSFORMER_V1"
    tokenizer: str = "ICMAT_GREEDY_PIECE_2048_V1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parameter_count"] = parameter_count(self)
        return value


def config_for_candidate(candidate_id: str) -> NanoLMConfig:
    if not candidate_id.startswith("CAND-G-"):
        raise ValueError(f"not a generative candidate: {candidate_id}")
    number = int(candidate_id.rsplit("-", 1)[1])
    if 1 <= number <= 6:
        return NanoLMConfig(
            family="FOUNDATION",
            vocab_size=VOCAB_SIZE,
            context_tokens=CONTEXT_TOKENS,
            max_generation_tokens=MAX_GENERATION_TOKENS,
            d_model=160,
            n_heads=5,
            n_layers=6,
            d_ff=320,
            parameter_min=800_000,
            parameter_max=1_800_000,
        )
    if 7 <= number <= 26:
        return NanoLMConfig(
            family="EXPERT",
            vocab_size=VOCAB_SIZE,
            context_tokens=CONTEXT_TOKENS,
            max_generation_tokens=MAX_GENERATION_TOKENS,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=256,
            parameter_min=400_000,
            parameter_max=1_200_000,
        )
    if 27 <= number <= 30:
        return NanoLMConfig(
            family="COLLABORATION",
            vocab_size=VOCAB_SIZE,
            context_tokens=CONTEXT_TOKENS,
            max_generation_tokens=MAX_GENERATION_TOKENS,
            d_model=112,
            n_heads=4,
            n_layers=4,
            d_ff=224,
            parameter_min=400_000,
            parameter_max=800_000,
        )
    raise ValueError(f"candidate outside the 30-model nano-LM contract: {candidate_id}")


def parameter_count(config: NanoLMConfig) -> int:
    """Exact trainable parameter count for the tied-embedding model."""

    d = config.d_model
    ff = config.d_ff
    # qkv/out weights, two FFN weights, their biases and two LayerNorms.
    per_block = 4 * d * d + 2 * d * ff + 9 * d + ff
    return (
        config.vocab_size * d
        + config.context_tokens * d
        + config.n_layers * per_block
        + 2 * d
    )


def validate_config(config: NanoLMConfig) -> None:
    count = parameter_count(config)
    if config.d_model % config.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if not config.parameter_min <= count <= config.parameter_max:
        raise ValueError(
            f"{config.family} parameter gate: {count} not in "
            f"[{config.parameter_min}, {config.parameter_max}]"
        )
    if config.quantization != "W8":
        raise ValueError("nano-LM formal route is W8 only")


def build_model(torch: Any, nn: Any, config: NanoLMConfig) -> Any:
    """Build the fixed-op causal Transformer used by the CUDA trainer."""

    validate_config(config)

    class CausalBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d = config.d_model
            self.norm1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d)
            self.ff1 = nn.Linear(d, config.d_ff)
            self.ff2 = nn.Linear(config.d_ff, d)
            self.scale = (d // config.n_heads) ** -0.5

        def forward(self, value: Any, causal_mask: Any) -> Any:
            batch, length, width = value.shape
            head_width = width // config.n_heads
            normalized = self.norm1(value)
            qkv = self.qkv(normalized).reshape(
                batch, length, 3, config.n_heads, head_width
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(causal_mask, -10_000.0)
            attention = torch.softmax(scores, dim=-1)
            attended = torch.matmul(attention, v).transpose(1, 2).reshape(
                batch, length, width
            )
            value = value + self.proj(attended)
            hidden = self.ff2(torch.relu(self.ff1(self.norm2(value))))
            return value + hidden

    class NanoTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(
                config.vocab_size, config.d_model, padding_idx=0
            )
            self.position_embedding = nn.Embedding(
                config.context_tokens, config.d_model
            )
            self.blocks = nn.ModuleList(CausalBlock() for _ in range(config.n_layers))
            self.final_norm = nn.LayerNorm(config.d_model)
            self.register_buffer(
                "causal_mask",
                torch.triu(
                    torch.ones(
                        config.context_tokens,
                        config.context_tokens,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                ).reshape(1, 1, config.context_tokens, config.context_tokens),
                persistent=False,
            )

        def forward(self, tokens: Any, return_hidden: bool = False) -> Any:
            length = tokens.shape[1]
            if length > config.context_tokens:
                raise ValueError("sequence exceeds frozen nano-LM context")
            positions = torch.arange(length, device=tokens.device)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
            mask = self.causal_mask[:, :, :length, :length]
            for block in self.blocks:
                hidden = block(hidden, mask)
            hidden = self.final_norm(hidden)
            logits = torch.nn.functional.linear(
                hidden, self.token_embedding.weight
            )
            return (logits, hidden) if return_hidden else logits

    model = NanoTransformerLM()
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = parameter_count(config)
    if actual != expected:
        raise RuntimeError(f"nano-LM parameter accounting mismatch: {actual} != {expected}")
    return model


for _candidate in ("CAND-G-001", "CAND-G-007", "CAND-G-027"):
    validate_config(config_for_candidate(_candidate))
