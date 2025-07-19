from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvConfig:
    device_aux: str = "cuda"
    int8: bool = False
    half: bool = True

@dataclass
class NeighborhoodConfig:
    model: str = "t5-base"
    random_fills: bool = False
    random_fills_tokens: bool = False
    pct_words_masked: float = 0.3
    span_length: int = 3
    buffer_size: int = 1
    top_p: float = 0.9
    max_tries: int = 3
    chunk_size: int = 16
    ceil_pct: bool = True
    neighbor_strategy: str = "deterministic"
    pct_swap_bert: float = 0.2
    original_tokenization_swap: bool = True

@dataclass
class ExperimentConfig:
    pretokenized: bool = False
    max_tokens: int = 512
    chunk_size: int = 16
    env_config: EnvConfig = EnvConfig()
    neighborhood_config: NeighborhoodConfig = NeighborhoodConfig()
