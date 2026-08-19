from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import LogEvent, ParsedLog, SignalNode


MAX_EVENTS_PER_SIGNAL = 200


@dataclass(frozen=True)
class EventDetectionSummary:
    """Counts describing event candidates discarded by detection limits."""

    total_candidates: int
    per_signal_truncated_count: int
    global_truncated_count: int
    returned_count: int

    @property
    def truncated_count(self) -> int:
        return self.per_signal_truncated_count + self.global_truncated_count

    @property
    def truncated(self) -> bool:
        return self.truncated_count > 0


@dataclass(frozen=True)
class EventDetectionResult:
    events: list[LogEvent]
    summary: EventDetectionSummary


def _normalize_token(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _event_type_for_signal(signal: SignalNode) -> str | None:
    tokens = [signal.name, signal.full_path, *signal.path_parts]
    if any(_normalize_token(token) == "errcode" for token in tokens):
        return "报警/错误变化"
    return None


def _changed_indices(values: np.ndarray, event_type: str) -> np.ndarray:
    changed, _ = _changed_indices_with_stats(values, event_type)
    return changed


def _changed_indices_with_stats(values: np.ndarray, event_type: str) -> tuple[np.ndarray, int]:
    finite_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    changed = np.flatnonzero(finite_pairs & (values[1:] != values[:-1])) + 1
    if event_type == "报警/错误变化":
        changed = changed[(values[changed] != 0) | (values[changed - 1] != 0)]
    limited = changed[:MAX_EVENTS_PER_SIGNAL]
    return limited, max(0, int(changed.size - limited.size))


def detect_events_result(parsed_log: ParsedLog, *, max_events: int = 1000) -> EventDetectionResult:
    events: list[LogEvent] = []
    total_candidates = 0
    per_signal_truncated_count = 0

    for signal in parsed_log.signals:
        if not signal.available or signal.signal_id not in parsed_log.signals_by_id:
            continue

        event_type = _event_type_for_signal(signal)
        if event_type is None:
            continue

        values = parsed_log.get_series(signal.signal_id)
        if values.shape[0] < 2:
            continue

        changed_indices, signal_truncated_count = _changed_indices_with_stats(values, event_type)
        per_signal_truncated_count += signal_truncated_count
        total_candidates += int(changed_indices.size + signal_truncated_count)
        for index in changed_indices:
            sample_index = int(index)
            events.append(
                LogEvent(
                    sample_index=sample_index,
                    time_seconds=float(parsed_log.time_seconds[sample_index]),
                    time_raw=str(parsed_log.time_raw[sample_index]),
                    signal_id=signal.signal_id,
                    signal_name=signal.name,
                    signal_path=signal.full_path,
                    previous_value=float(values[sample_index - 1]),
                    current_value=float(values[sample_index]),
                    event_type=event_type,
                )
            )

    ordered_events = sorted(events, key=lambda event: (event.sample_index, event.signal_path))
    limited_events = ordered_events[:max_events]
    global_truncated_count = max(0, len(ordered_events) - len(limited_events))
    summary = EventDetectionSummary(
        total_candidates=total_candidates,
        per_signal_truncated_count=per_signal_truncated_count,
        global_truncated_count=global_truncated_count,
        returned_count=len(limited_events),
    )
    return EventDetectionResult(events=limited_events, summary=summary)


def detect_events(parsed_log: ParsedLog, *, max_events: int = 1000) -> list[LogEvent]:
    """Return detected events while preserving the original list-returning API."""
    return list(detect_events_result(parsed_log, max_events=max_events).events)
