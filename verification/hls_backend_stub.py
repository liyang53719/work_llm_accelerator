from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import torch

from backend_interface import BackendInterface, DecodeResult, PrefillResult


class HlsBackendStub(BackendInterface):
    def __init__(self, prefill_lib: str | None = None, decode_lib: str | None = None) -> None:
        self.prefill_lib_path = Path(prefill_lib) if prefill_lib is not None else None
        self.decode_lib_path = Path(decode_lib) if decode_lib is not None else None
        self._prefill_lib = None
        self._decode_lib = None

        if self.prefill_lib_path is not None and self.prefill_lib_path.exists():
            self._prefill_lib = ctypes.CDLL(str(self.prefill_lib_path))
        if self.decode_lib_path is not None and self.decode_lib_path.exists():
            self._decode_lib = ctypes.CDLL(str(self.decode_lib_path))

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        raise NotImplementedError(
            "HLS full-model backend is not implemented yet. "
            "Current shared-library stubs only validate ABI shape, not Qwen2.5-1.5B math correctness."
        )

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        raise NotImplementedError(
            "HLS full-model backend is not implemented yet. "
            "Current shared-library stubs only validate ABI shape, not Qwen2.5-1.5B math correctness."
        )