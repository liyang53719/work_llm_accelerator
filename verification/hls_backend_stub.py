from __future__ import annotations

from typing import Any

import torch

from backend_interface import BackendInterface, DecodeResult, PrefillResult


class HlsBackendStub(BackendInterface):
    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        raise NotImplementedError("HLS backend wrapper is not implemented yet.")

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        raise NotImplementedError("HLS backend wrapper is not implemented yet.")