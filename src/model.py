"""
Chess Transformer Model for the Chess Challenge.

This module provides a simple GPT-style transformer architecture
designed to fit within the 1M parameter constraint.

Key components:
- ChessConfig: Configuration class for model hyperparameters
- ChessForCausalLM: The main model class for next-move prediction
"""

from __future__ import annotations

from pprint import pformat
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


class ChessConfig(PretrainedConfig):
    """
    Configuration class for the Chess Transformer model.

    This configuration is designed for a ~1M parameter model.
    Students can adjust these values to explore different architectures.

    Parameter budget breakdown (with default values):
    - Embeddings (vocab): 1200 x 128 = 153,600
    - Position Embeddings: 256 x 128 = 32,768
    - Transformer Layers: 6 x ~120,000 = ~720,000
    - LM Head (with weight tying): 0 (shared with embeddings)
    - Total: ~906,000 parameters

    Attributes:
        vocab_size: Size of the vocabulary (number of unique moves).
        n_embd: Embedding dimension (d_model).
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_ctx: Maximum sequence length (context window).
        n_inner: Feed-forward inner dimension (default: 3 * n_embd).
        dropout: Dropout probability.
        layer_norm_epsilon: Epsilon for layer normalization.
        tie_weights: Whether to tie embedding and output weights.
    """

    model_type = "chess_transformer"

    def __init__(
        self,
        vocab_size: int = 1200,
        n_embd: int = 256,
        n_layer: int = 10,
        n_head_kv: int = 8,
        n_head_q_per_kv: int = 2,
        dim_head_qk: int = 32,
        dim_head_v: Optional[int] = None,
        n_ctx: int = 1024,
        n_inner: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        tie_weights: bool = True,
        rope_theta: float = 1e4,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.dim_head_qk = dim_head_qk
        self.dim_head_v = dim_head_v or dim_head_qk

        self.n_head_kv = n_head_kv
        self.n_head_q_per_kv = n_head_q_per_kv

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head_kv = n_head_kv
        self.n_ctx = n_ctx
        self.n_inner = n_inner if n_inner is not None else 3 * n_embd  # Reduced from 4x to 3x
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_weights = tie_weights
        self.rope_theta = rope_theta
        # Inform HF base class about tying behavior
        self.tie_word_embeddings = bool(tie_weights)

    @property
    def dim_q(self):
        return self.n_head_q * self.dim_head_qk

    @property
    def dim_k(self):
        return self.n_head_kv * self.dim_head_qk

    @property
    def dim_v(self):
        return self.n_head_kv * self.dim_head_v

    @property
    def n_head_q(self):
        return self.n_head_q_per_kv * self.n_head_kv

    def __repr__(self):
        cls = self.__class__.__name__
        fields = self.to_dict()
        return f"{cls}(\n{pformat(fields, indent=2)}\n)"

    __str__ = __repr__


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary embeddings to input tensor x."""
    # Reshape x to complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)
    # Perform rotation in complex space
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    This is a standard scaled dot-product attention implementation
    with causal masking for autoregressive generation.
    """

    bias: torch.Tensor  # to restrict type to Tensor and not Module

    def __init__(self, config: ChessConfig):
        super().__init__()

        self._config = config

        self.proj_q = nn.Linear(config.n_embd, self.dim_q)
        self.proj_k = nn.Linear(config.n_embd, self.dim_k)
        self.proj_v = nn.Linear(config.n_embd, self.dim_v)

        self.proj_o = nn.Linear(self._n_head_q * self._dim_head_v, config.n_embd)

        # Causal mask (will be created on first forward pass)
        self.register_buffer(
            "bias",
            torch.ones(config.n_ctx, config.n_ctx, dtype=torch.bool)
            .tril(diagonal=0)
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False,
        )

    @property
    def dim_q(self):
        return self._config.dim_q

    @property
    def dim_k(self):
        return self._config.dim_k

    @property
    def dim_v(self):
        return self._config.dim_v

    @property
    def enable_gqa(self):
        return self._n_head_q_per_kv > 1

    @property
    def dropout_p(self):
        return self._config.dropout * self.training

    @property
    def _n_head_kv(self):
        return self._config.n_head_kv

    @property
    def _n_head_q(self):
        return self._config.n_head_q

    @property
    def _dim_head_qk(self):
        return self._config.dim_head_qk

    @property
    def _dim_head_v(self):
        return self._config.dim_head_v

    @property
    def _n_head_q_per_kv(self):
        return self._config.n_head_q_per_kv

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        q, k, v = (proj(x) for proj in (self.proj_q, self.proj_k, self.proj_v))

        # Reshape for multi-head attention
        q = q.unflatten(-1, (self._n_head_q, self._dim_head_qk))
        k = k.unflatten(-1, (self._n_head_kv, self._dim_head_qk))
        v = v.unflatten(-1, (self._n_head_kv, self._dim_head_v))

        q, k = (apply_rotary_emb(x, freqs_cis) for x in (q, k))

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        attn_mask = self.bias[..., :seq_len, :seq_len]

        # merge causal mask with attention mask if provided
        if attention_mask is not None:
            attention_mask = (
                attention_mask.view(batch_size, 1, 1, seq_len)
                .expand(-1, -1, seq_len, -1)
                .to(torch.bool)
            )
            attn_mask = torch.logical_or(attention_mask, attn_mask)

        attn_output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                enable_gqa=self.enable_gqa,
            )
            .transpose(1, 2)
            .flatten(2)
        )

        return self.proj_o(attn_output)


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) module.

    Standard two-layer MLP with GELU activation.
    """

    def __init__(self, config: ChessConfig):
        super().__init__()

        self.proj_up = nn.Linear(config.n_embd, config.n_inner)
        self.proj_down = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_up(x)
        x = F.gelu(x)
        x = self.proj_down(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block with attention and feed-forward layers.

    Uses pre-normalization (LayerNorm before attention/FFN) for better
    training stability.
    """

    def __init__(self, config: ChessConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis, attention_mask=attention_mask)
        # Pre-norm FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class ChessForCausalLM(PreTrainedModel):
    """
    Chess Transformer for Causal Language Modeling (next-move prediction).

    This model is designed to predict the next chess move given a sequence
    of previous moves. It uses a GPT-style architecture with:
    - Token embeddings for chess moves
    - Learned positional embeddings
    - Stacked transformer blocks
    - Linear head for next-token prediction

    The model supports weight tying between the embedding layer and the
    output projection to save parameters.

    Example:
        >>> config = ChessConfig(vocab_size=1200, n_embd=128, n_layer=6)
        >>> model = ChessForCausalLM(config)
        >>> inputs = {"input_ids": torch.tensor([[1, 42, 87]])}
        >>> outputs = model(**inputs)
        >>> next_move_logits = outputs.logits[:, -1, :]
    """

    config_class = ChessConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    # Suppress missing-key warning for tied lm_head when loading
    keys_to_ignore_on_load_missing = ["lm_head.weight"]
    freqs_cis: torch.Tensor

    def __init__(self, config: ChessConfig):
        super().__init__(config)

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        freqs_cis = self._precompute_freqs_cis(config.dim_head_qk, config.n_ctx, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Declare tied weights for proper serialization
        if config.tie_weights:
            self._tied_weights_keys = ["lm_head.weight"]

        # Initialize weights
        self.post_init()

        # Tie weights if configured
        if config.tie_weights:
            self.tie_weights()

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, value: nn.Module):
        self.wte = value
        if getattr(self.config, "tie_weights", False):
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def tie_weights(self):
        # Use HF helper to tie or clone depending on config
        if getattr(self.config, "tie_weights", False) or getattr(
            self.config, "tie_word_embeddings", False
        ):
            self._tie_or_clone_weights(self.lm_head, self.wte)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            labels: Labels for language modeling loss.
            return_dict: Whether to return a ModelOutput object.

        Returns:
            CausalLMOutputWithPast containing loss (if labels provided) and logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()

        # Get embeddings
        hidden_states = self.drop(self.wte(input_ids))

        freqs_cis = self.freqs_cis[:seq_len]

        # Pass through transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, freqs_cis=freqs_cis, attention_mask=attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            ignore_index = self.config.pad_token_id or -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def generate_move(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> int:
        """
        Generate the next move given a sequence of moves.

        Args:
            input_ids: Token IDs of shape (1, seq_len).
            temperature: Sampling temperature (1.0 = no change).
            top_k: If set, only sample from top k tokens.
            top_p: If set, use nucleus sampling with this threshold.

        Returns:
            The token ID of the predicted next move.
        """
        self.eval()

        # Get logits for the last position
        outputs = self(input_ids)
        logits = outputs.logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())
