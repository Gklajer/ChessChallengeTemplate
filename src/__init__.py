"""Chess Challenge source module."""

from transformers.models import AutoConfig, AutoModelForCausalLM

from model import ChessConfig, ChessForCausalLM
from tokenizer import ChessTokenizer


# Lazy import for evaluate to avoid RuntimeWarning when running as module
def __getattr__(name):
    if name == "ChessEvaluator":
        from evaluate import ChessEvaluator

        return ChessEvaluator
    if name == "load_model_from_hub":
        from evaluate import load_model_from_hub

        return load_model_from_hub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Register the model with Auto classes for easy loading
AutoConfig.register("chess_transformer", ChessConfig)
AutoModelForCausalLM.register(ChessConfig, ChessForCausalLM)

__all__ = [
    "ChessConfig",
    "ChessForCausalLM",
    "ChessTokenizer",
    "ChessEvaluator",
    "load_model_from_hub",
]
