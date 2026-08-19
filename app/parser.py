from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np

from .models import LogFileMeta, ParsedLog, SignalNode

ProgressCallback = Callable[[int, str], bool]


class ParseError(ValueError):
    """Raised when the input file does not match the expected log format."""


@dataclass(frozen=True)
class _LeafDefinition:
    signal_id: str
    name: str
    path_parts: tuple[str, ...]

    @property
    def full_path(self) -> str:
        return " / ".join(self.path_parts)


def _split_tab_fields(line: str) -> list[str]:
    fields = line.rstrip("\r\n").split("\t")
    while fields and fields[-1] == "":
        fields.pop()
    return fields


def _parse_time_token(token: str) -> datetime:
    token = token.strip()
    if len(token) != 17 or not token.isdigit():
        raise ValueError(f"invalid timestamp: {token}")

    base = datetime.strptime(token[:14], "%Y%m%d%H%M%S")
    milliseconds = int(token[14:])
    return base.replace(microsecond=milliseconds * 1000)


def _walk_leaf_definitions(element: ET.Element, path_parts: tuple[str, ...]) -> Iterable[_LeafDefinition]:
    tag = element.tag
    next_path = path_parts

    if tag in {"group", "dev", "puts"}:
        name = (element.attrib.get("name") or "").strip()
        if name:
            next_path = (*path_parts, name)

    if tag == "put":
        signal_id = (element.attrib.get("id") or "").strip()
        name = (element.text or "").strip()
        if signal_id and name:
            yield _LeafDefinition(
                signal_id=signal_id,
                name=name,
                path_parts=(*path_parts, name),
            )
        return

    for child in element:
        yield from _walk_leaf_definitions(child, next_path)


def build_signal_tree(xml_section: str, id_row: str) -> list[SignalNode]:
    if not xml_section.strip():
        raise ParseError("文件缺少 XML 头。")

    id_fields = _split_tab_fields(id_row)
    if not id_fields or id_fields[0] != "ID":
        raise ParseError("未找到有效的 ID 表头。")

    column_ids = id_fields[1:]
    non_empty_column_ids = [signal_id for signal_id in column_ids if signal_id]
    if len(non_empty_column_ids) != len(set(non_empty_column_ids)):
        raise ParseError("ID 表头中存在重复信号 ID。")
    id_to_index = {signal_id: index for index, signal_id in enumerate(column_ids)}

    try:
        root = ET.fromstring(xml_section)
    except ET.ParseError as exc:
        raise ParseError(f"XML 头解析失败: {exc}") from exc

    xml_put_ids = [
        (element.attrib.get("id") or "").strip()
        for element in root.iter("put")
    ]
    non_empty_xml_put_ids = [signal_id for signal_id in xml_put_ids if signal_id]
    if len(non_empty_xml_put_ids) != len(set(non_empty_xml_put_ids)):
        raise ParseError("XML 头中存在重复信号 ID。")

    leaf_definitions = list(_walk_leaf_definitions(root, ()))
    if not leaf_definitions:
        raise ParseError("XML 头中未找到任何信号定义。")

    signal_nodes: list[SignalNode] = []
    defined_ids: set[str] = set()

    for leaf in leaf_definitions:
        defined_ids.add(leaf.signal_id)
        column_index = id_to_index.get(leaf.signal_id)
        signal_nodes.append(
            SignalNode(
                signal_id=leaf.signal_id,
                name=leaf.name,
                path_parts=leaf.path_parts,
                full_path=leaf.full_path,
                column_index=column_index,
                available=column_index is not None,
            )
        )

    for index, signal_id in enumerate(column_ids):
        if signal_id in defined_ids:
            continue
        path_parts = ("Unknown", signal_id)
        signal_nodes.append(
            SignalNode(
                signal_id=signal_id,
                name=signal_id,
                path_parts=path_parts,
                full_path=" / ".join(path_parts),
                column_index=index,
                available=True,
                is_unknown=True,
            )
        )

    return sorted(
        signal_nodes,
        key=lambda node: (
            node.column_index is None,
            node.column_index if node.column_index is not None else 10**9,
            node.full_path,
        ),
    )


def _report_progress(callback: ProgressCallback | None, percent: int, message: str) -> None:
    if callback is not None and not callback(percent, message):
        raise ParseError("用户取消了文件加载。")


def parse_log_file(path: str | Path, progress_callback: ProgressCallback | None = None) -> ParsedLog:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise ParseError(f"文件不存在: {file_path}")
    if not file_path.is_file():
        raise ParseError(f"不是有效文件: {file_path}")

    xml_lines: list[str] = []
    signal_nodes: list[SignalNode] | None = None
    column_ids: list[str] | None = None
    expected_fields: int | None = None
    separator_found = False
    skipped_rows = 0
    time_raw: list[str] = []
    time_seconds: list[float] = []
    row_values_buffer = array("d")
    start_time: datetime | None = None
    previous_time: datetime | None = None
    total_bytes = max(file_path.stat().st_size, 1)
    line_count = 0

    _report_progress(progress_callback, 0, "正在读取文件...")
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        phase = "xml"
        while True:
            raw_line = handle.readline()
            if raw_line == "":
                break

            line_count += 1
            if line_count % 256 == 0:
                percent = min(95, int((handle.tell() / total_bytes) * 95))
                _report_progress(progress_callback, percent, "正在解析数据...")

            stripped = raw_line.strip()

            if phase == "xml":
                if stripped and set(stripped) == {"*"}:
                    separator_found = True
                    phase = "id"
                    continue
                xml_lines.append(raw_line)
                continue

            if phase == "id":
                if not stripped:
                    continue
                signal_nodes = build_signal_tree("".join(xml_lines), raw_line)
                id_fields = _split_tab_fields(raw_line)
                column_ids = id_fields[1:]
                expected_fields = len(id_fields)
                phase = "data"
                continue

            if not stripped:
                continue

            if expected_fields is None or column_ids is None:
                raise ParseError("内部错误: 数据读取前未初始化表头。")

            fields = _split_tab_fields(raw_line)
            if len(fields) != expected_fields:
                skipped_rows += 1
                continue

            try:
                row_time = _parse_time_token(fields[0])
                row_values = np.fromiter((float(value) for value in fields[1:]), dtype=float, count=len(column_ids))
            except (TypeError, ValueError, OverflowError):
                skipped_rows += 1
                continue

            if row_values.shape[0] != len(column_ids):
                skipped_rows += 1
                continue

            # Non-finite samples and backwards timestamps break interpolation/searchsorted assumptions.
            if not np.isfinite(row_values).all() or (previous_time is not None and row_time < previous_time):
                skipped_rows += 1
                continue

            if start_time is None:
                start_time = row_time
            previous_time = row_time

            time_raw.append(fields[0])
            time_seconds.append((row_time - start_time).total_seconds())
            row_values_buffer.extend(row_values)

    _report_progress(progress_callback, 96, "正在整理曲线数据...")
    if not separator_found:
        raise ParseError("文件缺少 XML 与数据区的分隔线。")
    if signal_nodes is None or column_ids is None or expected_fields is None:
        raise ParseError("未找到 ID 表头。")
    if not time_raw:
        raise ParseError("未读取到任何有效数据行。")

    matrix = np.frombuffer(row_values_buffer, dtype=np.float64).reshape(len(time_raw), len(column_ids))
    signals_by_id = {signal_id: matrix[:, index] for index, signal_id in enumerate(column_ids)}

    meta = LogFileMeta(
        path=file_path,
        sample_count=len(time_raw),
        field_count=len(column_ids),
        start_time_raw=time_raw[0],
        end_time_raw=time_raw[-1],
        skipped_rows=skipped_rows,
    )

    parsed = ParsedLog(
        meta=meta,
        time_raw=np.array(time_raw, dtype=str),
        time_seconds=np.array(time_seconds, dtype=float),
        signals_by_id=signals_by_id,
        signals=signal_nodes,
        skipped_rows=skipped_rows,
    )
    _report_progress(progress_callback, 98, "曲线数据已加载")
    return parsed
