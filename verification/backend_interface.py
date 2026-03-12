from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


CacheType = Any


@dataclass
class PrefillResult:
    logits: torch.Tensor
    cache: CacheType


@dataclass
class DecodeResult:
    logits: torch.Tensor
    cache: CacheType


class BackendInterface(ABC):
    @abstractmethod
    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        raise NotImplementedError

    @abstractmethod
    def decode_step(self, input_ids: torch.Tensor, cache: CacheType) -> DecodeResult:
        raise NotImplementedError