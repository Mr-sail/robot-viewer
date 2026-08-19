from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LogFileMeta:
    path: Path
    sample_count: int
    field_count: int
    start_time_raw: str
    end_time_raw: str
    skipped_rows: int = 0


@dataclass(frozen=True)
class SignalNode:
    signal_id: str
    name: str
    path_parts: tuple[str, ...]
    full_path: str
    column_index: int | None
    available: bool
    is_unknown: bool = False


@dataclass
class ParsedLog:
    meta: LogFileMeta
    time_raw: np.ndarray
    time_seconds: np.ndarray
    signals_by_id: dict[str, np.ndarray]
    signals: list[SignalNode]
    skipped_rows: int = 0

    def get_series(self, signal_id: str) -> np.ndarray:
        return self.signals_by_id[signal_id]


@dataclass(frozen=True)
class LogEvent:
    sample_index: int
    time_seconds: float
    time_raw: str
    signal_id: str
    signal_name: str
    signal_path: str
    previous_value: float
    current_value: float
    event_type: str
