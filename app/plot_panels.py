from __future__ import annotations

from collections.abc import Callable
import html
from pathlib import Path
import re

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .mesh_loader import LoadedMesh, MeshLoadError, load_mesh
from .models import ParsedLog, SignalNode
from .robot_model import RobotModel, RobotModelError, load_robot_model

try:
    import pyqtgraph.opengl as gl

    GL_AVAILABLE = True
    GL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on optional local dependency
    gl = None
    GL_AVAILABLE = False
    GL_IMPORT_ERROR = str(exc)


ROBOT_LINK_COLOR = (0.09, 0.43, 0.86, 1.0)
ROBOT_JOINT_COLOR = (0.98, 0.62, 0.17, 1.0)
ROBOT_MESH_COLOR = (0.52, 0.62, 0.72, 0.58)
ROBOT_MESH_EDGE_COLOR = (0.18, 0.24, 0.32, 0.22)
AXIS_X_COLOR = (0.89, 0.16, 0.16, 0.9)
AXIS_Y_COLOR = (0.10, 0.64, 0.23, 0.9)
AXIS_Z_COLOR = (0.13, 0.44, 0.82, 0.9)
WORLD_AXIS_X_COLOR = (0.89, 0.16, 0.16, 0.45)
WORLD_AXIS_Y_COLOR = (0.10, 0.64, 0.23, 0.45)
WORLD_AXIS_Z_COLOR = (0.13, 0.44, 0.82, 0.45)


class AxisZoomPlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._zoom_mode = "auto"
        self._default_zoom_mode = "xy"

    def set_zoom_mode(self, mode: str) -> None:
        self._zoom_mode = mode

    def set_default_zoom_mode(self, mode: str) -> None:
        self._default_zoom_mode = mode

    def _effective_zoom_mode(self, scene_pos) -> str:
        if self._zoom_mode in {"x", "y", "xy"}:
            return self._zoom_mode

        plot_item = self.getPlotItem()
        view_box = plot_item.vb
        if view_box.sceneBoundingRect().contains(scene_pos):
            return self._default_zoom_mode

        left_axis = plot_item.getAxis("left")
        bottom_axis = plot_item.getAxis("bottom")
        if left_axis.sceneBoundingRect().contains(scene_pos):
            return "y"
        if bottom_axis.sceneBoundingRect().contains(scene_pos):
            return "x"
        return self._default_zoom_mode

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)

        scene_pos = self.mapToScene(event.position().toPoint())
        zoom_mode = self._effective_zoom_mode(scene_pos)
        if zoom_mode not in {"x", "y", "xy"}:
            return super().wheelEvent(event)
        mouse_point = self.getPlotItem().vb.mapSceneToView(scene_pos)
        factor = 0.85 if delta > 0 else 1.0 / 0.85
        x_factor = factor if zoom_mode in {"x", "xy"} else 1.0
        y_factor = factor if zoom_mode in {"y", "xy"} else 1.0
        self.getPlotItem().vb.scaleBy(x=x_factor, y=y_factor, center=mouse_point)
        event.accept()


class BasePlotPanel(QFrame):
    def __init__(
        self,
        mode: str,
        activate_callback: Callable[["BasePlotPanel"], None],
        remove_callback: Callable[["BasePlotPanel"], None],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.selected_signal_ids: list[str] = []
        self.time_range: tuple[float, float] | None = None
        self._activate_callback = activate_callback
        self._remove_callback = remove_callback
        self._status_callback: Callable[[str], None] | None = None

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("plotPanel")
        self.setStyleSheet(
            """
            QFrame#plotPanel {
                border: 2px solid #d9d9d9;
                border-radius: 8px;
                background: #ffffff;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        self.header_layout = header_layout
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("color: #666666;")
        self.active_label = QLabel("当前图")
        self.active_label.setStyleSheet("color: #1f6feb; font-weight: 700;")
        self.active_label.hide()
        self.activate_button = QPushButton("设为当前图")
        self.remove_button = QPushButton("关闭此图")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.detail_label, 1)
        header_layout.addWidget(self.active_label)
        header_layout.addWidget(self.activate_button)
        header_layout.addWidget(self.remove_button)
        layout.addLayout(header_layout)

        self.message_label = QLabel("No data loaded")
        self.message_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.message_label)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        layout.addWidget(self.content_widget, 1)

        self.activate_button.clicked.connect(lambda: self._activate_callback(self))
        self.remove_button.clicked.connect(lambda: self._remove_callback(self))

    @property
    def selection_limit(self) -> int | None:
        return None

    @property
    def supports_time_range(self) -> bool:
        return True

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        self._status_callback = callback

    def emit_status(self, text: str) -> None:
        if self._status_callback is not None:
            self._status_callback(text)

    def set_panel_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_active(self, active: bool) -> None:
        if active:
            self.setStyleSheet(
                """
                QFrame#plotPanel {
                    border: 2px solid #1f6feb;
                    border-radius: 8px;
                    background: #ffffff;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#plotPanel {
                    border: 2px solid #d9d9d9;
                    border-radius: 8px;
                    background: #ffffff;
                }
                """
            )
        self.active_label.setVisible(active)
        self.activate_button.setVisible(not active)
        self.activate_button.setEnabled(not active)

    def set_zoom_mode(self, mode: str) -> None:
        del mode

    def set_time_range(self, start_seconds: float, end_seconds: float) -> None:
        self.time_range = (start_seconds, end_seconds)

    def clear_time_range(self) -> None:
        self.time_range = None

    def clamp_time_range(self, max_seconds: float) -> None:
        if self.time_range is None:
            return

        start_seconds, end_seconds = self.time_range
        start_seconds = max(0.0, min(start_seconds, max_seconds))
        end_seconds = max(0.0, min(end_seconds, max_seconds))
        self.time_range = (start_seconds, end_seconds) if start_seconds < end_seconds else None

    def _time_indices(self, parsed_log: ParsedLog) -> np.ndarray:
        sample_count = parsed_log.time_seconds.shape[0]
        if self.time_range is None:
            return np.arange(sample_count, dtype=int)

        start_seconds, end_seconds = self.time_range
        mask = (parsed_log.time_seconds >= start_seconds) & (parsed_log.time_seconds <= end_seconds)
        return np.flatnonzero(mask)

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        del parsed_log, signal_lookup

    def reset_view(self) -> None:
        return

    def sync_sample_index(self, index: int | None) -> None:
        del index

    def focus_sample_index(self, index: int) -> None:
        self.sync_sample_index(index)


class Plot2DPanel(BasePlotPanel):
    def __init__(
        self,
        mode: str,
        activate_callback: Callable[[BasePlotPanel], None],
        remove_callback: Callable[[BasePlotPanel], None],
    ) -> None:
        super().__init__(mode=mode, activate_callback=activate_callback, remove_callback=remove_callback)
        self.curves: list[pg.PlotDataItem] = []
        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self._default_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._cursor_sync_callback: Callable[[int | None], None] | None = None
        self._visible_indices: np.ndarray = np.array([], dtype=int)
        self._measurement_indices: list[int] = []
        self._measurement_status: str | None = None
        self._current_cursor_index: int | None = None
        self._zoom_mode = "auto"
        self._display_series_by_id: dict[str, np.ndarray] = {}
        self._xt_display_mode = "raw"
        self._xt_mode_group: QButtonGroup | None = None
        self._xt_mode_buttons: dict[str, QPushButton] = {}

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.plot_widget = AxisZoomPlotWidget(background="w")
        self.plot_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.getPlotItem().setDownsampling(mode="peak")
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.set_default_zoom_mode("xy")

        self.legend = None
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#555555", width=1))
        self.h_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#999999", width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.v_line.setZValue(20)
        self.h_line.setZValue(19)
        self._hide_cursor()

        self.content_layout.addWidget(self.plot_widget, 1)

        self.value_overlay = QLabel(self.plot_widget)
        self.value_overlay.setStyleSheet(
            """
            QLabel {
                background: rgba(255, 255, 255, 210);
                border: 1px solid rgba(31, 111, 235, 120);
                border-radius: 6px;
                padding: 6px 8px;
                color: #1f2937;
                font-size: 12px;
            }
            """
        )
        self.value_overlay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.value_overlay.setWordWrap(True)
        self.value_overlay.setMinimumWidth(220)
        self.value_overlay.setMaximumWidth(320)
        self._set_overlay_text("Values will appear here")
        self.value_overlay.raise_()
        self.value_overlay.show()

        self.equal_scale_button: QPushButton | None = None
        if self.mode == "xt":
            self._build_xt_mode_selector()
        if self.mode == "xy":
            self.equal_scale_button = QPushButton("1:1")
            self.equal_scale_button.setCheckable(True)
            self.equal_scale_button.setToolTip("Keep X and Y using the same display scale")
            self.equal_scale_button.toggled.connect(self._on_equal_scale_toggled)
            self.header_layout.insertWidget(2, self.equal_scale_button)

        self.plot_widget.installEventFilter(self)
        self.plot_widget.viewport().installEventFilter(self)

        self.mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        self.set_zoom_mode("auto")

    @property
    def selection_limit(self) -> int | None:
        if self.mode == "xy":
            return 2
        return None

    def _build_xt_mode_selector(self) -> None:
        mode_widget = QWidget(self)
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)

        self._xt_mode_group = QButtonGroup(self)
        self._xt_mode_group.setExclusive(True)

        button_specs = (
            ("raw", "Raw", "Plot the original signal values"),
            ("diff1", "D1", "Plot the first sample-to-sample difference"),
            ("diff2", "D2", "Plot the second sample-to-sample difference"),
        )
        for mode_key, label, tooltip in button_specs:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setToolTip(tooltip)
            button.setMinimumWidth(44)
            button.clicked.connect(lambda checked=False, selected=mode_key: self._set_xt_display_mode(selected))
            self._xt_mode_group.addButton(button)
            self._xt_mode_buttons[mode_key] = button
            mode_layout.addWidget(button)

        self._xt_mode_buttons[self._xt_display_mode].setChecked(True)
        self.header_layout.insertWidget(1, mode_widget)

    @staticmethod
    def _xt_mode_label(mode: str) -> str:
        labels = {
            "raw": "Raw",
            "diff1": "1st Difference",
            "diff2": "2nd Difference",
        }
        return labels.get(mode, "Raw")

    @staticmethod
    def _xt_mode_suffix(mode: str) -> str:
        suffixes = {
            "raw": "",
            "diff1": "D1",
            "diff2": "D2",
        }
        return suffixes.get(mode, "")

    def _set_xt_display_mode(self, mode: str) -> None:
        if self.mode != "xt":
            return
        if mode not in {"raw", "diff1", "diff2"}:
            return
        if mode == self._xt_display_mode:
            button = self._xt_mode_buttons.get(mode)
            if button is not None and not button.isChecked():
                button.setChecked(True)
            return

        self._xt_display_mode = mode
        button = self._xt_mode_buttons.get(mode)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._refresh_detail_label()

        if self.parsed_log is None:
            return

        current_index = self._current_cursor_index
        self.update_plot(self.parsed_log, self.signal_lookup)
        self.reset_view()
        if current_index is not None:
            self.sync_cursor_to_index(current_index)

    def _refresh_detail_label(self) -> None:
        if self._zoom_mode == "auto":
            default_mode = "X-T" if self.mode == "xt" else "X-Y"
            detail = f"Wheel: Auto | Hover axis to zoom X/Y | Plot area={default_mode}"
        else:
            detail = f"Wheel locked: {self._zoom_mode.upper()}"

        if self.mode == "xt":
            detail = f"{detail} | Series: {self._xt_mode_label(self._xt_display_mode)}"
        self.detail_label.setText(detail)

    def set_zoom_mode(self, mode: str) -> None:
        self._zoom_mode = mode
        self.plot_widget.set_zoom_mode(mode)
        self._refresh_detail_label()

    def set_cursor_sync_callback(self, callback: Callable[[int | None], None]) -> None:
        self._cursor_sync_callback = callback

    def _on_equal_scale_toggled(self, checked: bool) -> None:
        if self.mode != "xy":
            return
        self.plot_widget.getViewBox().setAspectLocked(lock=checked, ratio=1.0)
        if self._default_ranges is not None:
            self._apply_default_ranges()

    @staticmethod
    def _display_signal_label(signal: SignalNode) -> str:
        if len(signal.path_parts) > 1:
            return " / ".join(signal.path_parts[1:])
        return signal.full_path

    @staticmethod
    def _series_color(index: int) -> str:
        return MainPalette.colors[index % len(MainPalette.colors)]

    @staticmethod
    def _format_numeric_value(value: float) -> str:
        return f"{value:.6f}" if np.isfinite(value) else "n/a"

    def _format_overlay_rows(self, rows: list[tuple[str, float, str]]) -> str:
        html_rows = []
        for label, value, color in rows:
            safe_label = html.escape(label)
            html_rows.append(
                f"<span style='color:{color}; font-weight:600;'>{safe_label}</span>: "
                f"<span>{self._format_numeric_value(value)}</span>"
            )
        return "<br>".join(html_rows) if html_rows else "No values"

    def _clear_curves(self) -> None:
        for curve in self.curves:
            self.plot_widget.removeItem(curve)
        self.curves.clear()
        self._display_series_by_id.clear()
        if self.legend is not None:
            self.legend.clear()
        self._default_ranges = None
        self._current_cursor_index = None
        self._clear_measurement()
        self._hide_cursor()

    def _clear_measurement(self) -> None:
        self._measurement_indices.clear()
        self._measurement_status = None

    def _set_overlay_text(self, text: str) -> None:
        self.value_overlay.setText(text)
        self.value_overlay.adjustSize()
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        margin = 12
        x_pos = max(self.plot_widget.width() - self.value_overlay.width() - margin, margin)
        self.value_overlay.move(x_pos, margin)

    def _hide_cursor(self, *, broadcast: bool = False) -> None:
        self.v_line.hide()
        self.h_line.hide()
        if broadcast and self._cursor_sync_callback is not None:
            self._cursor_sync_callback(None)
        if hasattr(self, "plot_widget"):
            self.plot_widget.viewport().update()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.plot_widget and event.type() in {QEvent.Type.Resize, QEvent.Type.Show}:
            self._reposition_overlay()
        elif watched is self.plot_widget and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Left and self._step_cursor(-1):
                return True
            if event.key() == Qt.Key.Key_Right and self._step_cursor(1):
                return True
        elif watched is self.plot_widget.viewport() and event.type() in {QEvent.Type.Leave, QEvent.Type.Hide}:
            self._hide_cursor(broadcast=True)
        return super().eventFilter(watched, event)

    @staticmethod
    def _compute_padded_range(min_value: float, max_value: float) -> tuple[float, float]:
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            return -1.0, 1.0
        if min_value == max_value:
            padding = max(abs(min_value) * 0.05, 1.0)
            return min_value - padding, max_value + padding
        padding = (max_value - min_value) * 0.05
        return min_value - padding, max_value + padding

    @staticmethod
    def _build_xt_display_series(series: np.ndarray, mode: str) -> np.ndarray:
        values = np.asarray(series, dtype=float)
        if mode == "raw":
            return values
        if mode == "diff1":
            derived = np.full(values.shape, np.nan, dtype=float)
            if values.size >= 2:
                derived[1:] = np.diff(values)
            return derived
        if mode == "diff2":
            derived = np.full(values.shape, np.nan, dtype=float)
            if values.size >= 3:
                derived[2:] = np.diff(values, n=2)
            return derived
        raise ValueError(f"Unsupported X-T display mode: {mode}")

    def _get_display_series(self, signal_id: str) -> np.ndarray:
        cached = self._display_series_by_id.get(signal_id)
        if cached is not None:
            return cached

        assert self.parsed_log is not None
        series = self.parsed_log.get_series(signal_id)
        if self.mode == "xt":
            cached = self._build_xt_display_series(series, self._xt_display_mode)
        else:
            cached = np.asarray(series, dtype=float)
        self._display_series_by_id[signal_id] = cached
        return cached

    def _xt_display_signal_label(self, signal: SignalNode) -> str:
        label = self._display_signal_label(signal)
        suffix = self._xt_mode_suffix(self._xt_display_mode)
        return label if not suffix else f"{label} ({suffix})"

    def _set_default_ranges(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        x_finite = np.asarray(x_values, dtype=float)
        y_finite = np.asarray(y_values, dtype=float)
        x_finite = x_finite[np.isfinite(x_finite)]
        y_finite = y_finite[np.isfinite(y_finite)]

        if x_finite.size == 0:
            x_min, x_max = -1.0, 1.0
        else:
            x_min, x_max = self._compute_padded_range(float(np.min(x_finite)), float(np.max(x_finite)))

        if y_finite.size == 0:
            y_min, y_max = -1.0, 1.0
        else:
            y_min, y_max = self._compute_padded_range(float(np.min(y_finite)), float(np.max(y_finite)))
        self._default_ranges = ((x_min, x_max), (y_min, y_max))

    def _apply_default_ranges(self) -> None:
        if self._default_ranges is None:
            return
        (x_min, x_max), (y_min, y_max) = self._default_ranges
        self.plot_widget.setXRange(x_min, x_max, padding=0.0)
        self.plot_widget.setYRange(y_min, y_max, padding=0.0)

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        self.parsed_log = parsed_log
        self.signal_lookup = signal_lookup
        self._clear_curves()

        if parsed_log is None:
            self.message_label.setText("No data loaded")
            self._set_overlay_text("No data loaded")
            self._visible_indices = np.array([], dtype=int)
            return

        self._visible_indices = self._time_indices(parsed_log)
        if self.mode == "xt":
            self._update_xt_plot(parsed_log, signal_lookup)
        else:
            self._update_xy_plot(parsed_log, signal_lookup)

    def _update_xt_plot(self, parsed_log: ParsedLog, signal_lookup: dict[str, SignalNode]) -> None:
        self.plot_widget.setLabel("bottom", "Relative Time", units="s")
        self.plot_widget.setLabel("left", self._xt_mode_label(self._xt_display_mode))

        if not self.selected_signal_ids:
            self.message_label.setText("Select one or more signals to draw an X-T plot.")
            self._set_overlay_text("X-T plot is waiting for signal selection")
            return

        indices = self._visible_indices
        if indices.size == 0:
            self.message_label.setText("No samples in the selected time range.")
            self._set_overlay_text("Time range contains no samples")
            return

        time_values = parsed_log.time_seconds[indices]
        all_series: list[np.ndarray] = []
        for order, signal_id in enumerate(self.selected_signal_ids):
            if signal_id not in parsed_log.signals_by_id:
                continue
            signal = signal_lookup[signal_id]
            series = self._get_display_series(signal_id)[indices]
            curve = self.plot_widget.plot(
                time_values,
                series,
                pen=pg.mkPen(self._series_color(order), width=2),
                name=self._xt_display_signal_label(signal),
            )
            self.curves.append(curve)
            all_series.append(series)

        self.message_label.setText(
            f"X-T plot | {self._xt_mode_label(self._xt_display_mode)} | {len(self.curves)} signal(s)"
        )
        if all_series:
            combined = np.concatenate(all_series)
            self._set_default_ranges(time_values, combined)
            first_signal = signal_lookup[self.selected_signal_ids[0]]
            first_index = int(indices[0])
            first_value = float(self._get_display_series(self.selected_signal_ids[0])[first_index])
            self._set_overlay_text(
                self._format_overlay_rows(
                    [(self._xt_display_signal_label(first_signal), first_value, self._series_color(0))]
                )
            )

    def _update_xy_plot(self, parsed_log: ParsedLog, signal_lookup: dict[str, SignalNode]) -> None:
        if len(self.selected_signal_ids) != 2:
            self.plot_widget.setLabel("bottom", "X")
            self.plot_widget.setLabel("left", "Y")
            self.message_label.setText("Select exactly two signals for the current X-Y plot.")
            self._set_overlay_text("X-Y plot is waiting for 2 selected signals")
            return

        indices = self._visible_indices
        if indices.size == 0:
            self.message_label.setText("No samples in the selected time range.")
            self._set_overlay_text("Time range contains no samples")
            return

        x_signal_id, y_signal_id = self.selected_signal_ids
        x_signal = signal_lookup[x_signal_id]
        y_signal = signal_lookup[y_signal_id]
        x_series = self._get_display_series(x_signal_id)[indices]
        y_series = self._get_display_series(y_signal_id)[indices]
        finite_mask = np.isfinite(x_series) & np.isfinite(y_series)
        if not np.any(finite_mask):
            self._visible_indices = np.array([], dtype=int)
            self.message_label.setText("The selected X-Y signals contain no finite sample pairs.")
            self._set_overlay_text("No finite X-Y sample pairs")
            return
        self._visible_indices = indices[finite_mask]
        x_series = x_series[finite_mask]
        y_series = y_series[finite_mask]

        x_label = self._display_signal_label(x_signal)
        y_label = self._display_signal_label(y_signal)
        self.plot_widget.setLabel("bottom", x_label)
        self.plot_widget.setLabel("left", y_label)
        curve = self.plot_widget.plot(
            x_series,
            y_series,
            pen=pg.mkPen(MainPalette.colors[0], width=2),
            name=f"{y_label} vs {x_label}",
        )
        self.curves.append(curve)
        self.message_label.setText(f"X-Y plot: X={x_label} | Y={y_label}")
        self._set_default_ranges(x_series, y_series)
        self._set_overlay_text(
            self._format_overlay_rows(
                [
                    (f"X - {x_label}", float(x_series[0]), self._series_color(0)),
                    (f"Y - {y_label}", float(y_series[0]), self._series_color(1)),
                ]
            )
        )

    def reset_view(self) -> None:
        self._apply_default_ranges()

    def sync_sample_index(self, index: int | None) -> None:
        self.sync_cursor_to_index(index)

    def sync_cursor_to_index(self, index: int | None) -> None:
        if index is None or self.parsed_log is None or not self.curves:
            self._current_cursor_index = None
            self._hide_cursor()
            return

        sample_count = self.parsed_log.time_seconds.shape[0]
        if sample_count == 0:
            self._hide_cursor()
            return

        clamped_index = max(0, min(index, sample_count - 1))
        if self.time_range is not None and (
            self._visible_indices.size == 0
            or clamped_index < int(self._visible_indices[0])
            or clamped_index > int(self._visible_indices[-1])
        ):
            self._current_cursor_index = None
            self._hide_cursor()
            return

        self._current_cursor_index = clamped_index
        if self.mode == "xt":
            self._show_xt_cursor_at_index(clamped_index)
        else:
            self._show_xy_cursor_at_index(clamped_index)

    def focus_sample_index(self, index: int) -> None:
        if self.parsed_log is None or not self.curves:
            return

        if self.time_range is not None and (
            self._visible_indices.size == 0
            or index < int(self._visible_indices[0])
            or index > int(self._visible_indices[-1])
        ):
            self._hide_cursor()
            return

        self.sync_cursor_to_index(index)
        if self.mode != "xt":
            return

        times = self.parsed_log.time_seconds
        if times.shape[0] == 0:
            return

        clamped_index = max(0, min(index, times.shape[0] - 1))
        center = float(times[clamped_index])
        total_span = max(float(times[-1] - times[0]), 1.0)
        window = max(total_span * 0.02, 0.5)
        x_min = max(float(times[0]), center - window)
        x_max = min(float(times[-1]), center + window)
        if x_min < x_max:
            self.plot_widget.setXRange(x_min, x_max, padding=0.0)

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self.parsed_log is None or not self.curves:
            self.emit_status("No visible data in the current plot")
            self._hide_cursor(broadcast=True)
            return

        scene_pos = event[0]
        view_rect = self.plot_widget.getPlotItem().vb.sceneBoundingRect()
        if not view_rect.contains(scene_pos):
            self._hide_cursor(broadcast=True)
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)

        if self.mode == "xt":
            self._update_xt_cursor(mouse_point.x())
        else:
            self._update_xy_cursor(mouse_point.x(), mouse_point.y())

    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.plot_widget.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.parsed_log is None or not self.curves:
            return

        scene_pos = event.scenePos()
        view_rect = self.plot_widget.getPlotItem().vb.sceneBoundingRect()
        if not view_rect.contains(scene_pos):
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)
        index = self._nearest_sample_index(mouse_point.x(), mouse_point.y())
        if index is None:
            return

        self._current_cursor_index = index
        if self.mode == "xt":
            self._show_xt_cursor_at_index(index)
        else:
            self._show_xy_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)
        if len(self._measurement_indices) >= 2:
            self._measurement_indices.clear()

        self._measurement_indices.append(index)
        if len(self._measurement_indices) == 1:
            time_seconds = float(self.parsed_log.time_seconds[index])
            self._measurement_status = f"已选择第1点: t={time_seconds:.6f}s；再点击第2点计算时间差"
            self.emit_status(self._measurement_status)
            return

        first_index, second_index = self._measurement_indices
        first_time = float(self.parsed_log.time_seconds[first_index])
        second_time = float(self.parsed_log.time_seconds[second_index])
        delta_ms = abs(second_time - first_time) * 1000.0
        rounded_delta_ms = int(round(delta_ms))
        cycle_count = int(round(delta_ms / 4.0))
        self._measurement_status = (
            f"两点时间差: {rounded_delta_ms} ms | {cycle_count} 个周期(4ms/周期) | "
            f"点1 t={first_time:.6f}s, 点2 t={second_time:.6f}s"
        )
        self.emit_status(self._measurement_status)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Left:
            if self._step_cursor(-1):
                event.accept()
                return
        elif event.key() == Qt.Key.Key_Right:
            if self._step_cursor(1):
                event.accept()
                return

        super().keyPressEvent(event)

    def _step_cursor(self, direction: int) -> bool:
        if self.parsed_log is None or not self.curves or self._visible_indices.size == 0:
            return False

        if self._current_cursor_index is None:
            current_position = 0 if direction >= 0 else self._visible_indices.size - 1
        else:
            current_position = int(np.searchsorted(self._visible_indices, self._current_cursor_index))
            if current_position >= self._visible_indices.size:
                current_position = self._visible_indices.size - 1
            elif self._visible_indices[current_position] != self._current_cursor_index and direction < 0:
                current_position -= 1
            current_position = max(0, min(current_position + direction, self._visible_indices.size - 1))

        index = int(self._visible_indices[current_position])
        self._current_cursor_index = index
        self._clear_measurement()
        if self.mode == "xt":
            self._show_xt_cursor_at_index(index)
        else:
            self._show_xy_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)
        return True

    def _nearest_sample_index(self, x_value: float, y_value: float) -> int | None:
        if self.parsed_log is None or self._visible_indices.size == 0:
            return None

        if self.mode == "xt":
            times = self.parsed_log.time_seconds[self._visible_indices]
            local_index = int(np.searchsorted(times, x_value))
            if local_index >= len(times):
                local_index = len(times) - 1
            elif local_index > 0 and abs(times[local_index - 1] - x_value) <= abs(times[local_index] - x_value):
                local_index -= 1
            return int(self._visible_indices[local_index])

        if len(self.selected_signal_ids) != 2:
            return None

        x_series = self._get_display_series(self.selected_signal_ids[0])[self._visible_indices]
        y_series = self._get_display_series(self.selected_signal_ids[1])[self._visible_indices]
        local_index = self._nearest_xy_local_index(x_series, y_series, x_value, y_value)
        return None if local_index is None else int(self._visible_indices[local_index])

    @staticmethod
    def _nearest_xy_local_index(
        x_series: np.ndarray,
        y_series: np.ndarray,
        x_value: float,
        y_value: float,
    ) -> int | None:
        finite_mask = np.isfinite(x_series) & np.isfinite(y_series)
        finite_positions = np.flatnonzero(finite_mask)
        if finite_positions.size == 0:
            return None

        finite_x = x_series[finite_mask]
        finite_y = y_series[finite_mask]
        x_span = max(float(np.ptp(finite_x)), 1e-9)
        y_span = max(float(np.ptp(finite_y)), 1e-9)
        distances = ((finite_x - x_value) / x_span) ** 2 + ((finite_y - y_value) / y_span) ** 2
        return int(finite_positions[int(np.argmin(distances))])

    def _update_xt_cursor(self, x_value: float) -> None:
        assert self.parsed_log is not None
        if self._visible_indices.size == 0:
            self._hide_cursor(broadcast=True)
            return

        times = self.parsed_log.time_seconds[self._visible_indices]
        local_index = int(np.searchsorted(times, x_value))
        if local_index >= len(times):
            local_index = len(times) - 1
        elif local_index > 0 and abs(times[local_index - 1] - x_value) <= abs(times[local_index] - x_value):
            local_index -= 1
        index = int(self._visible_indices[local_index])
        self._current_cursor_index = index
        self._show_xt_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)

    def _show_xt_cursor_at_index(self, index: int) -> None:
        assert self.parsed_log is not None
        nearest_x = float(self.parsed_log.time_seconds[index])
        self.v_line.setPos(nearest_x)

        preview_parts = [self._xt_mode_suffix(self._xt_display_mode) or "Raw", f"t={nearest_x:.4f}s"]
        y_value = None
        for signal_id in self.selected_signal_ids[:3]:
            value = float(self._get_display_series(signal_id)[index])
            signal = self.signal_lookup[signal_id]
            preview_parts.append(f"{signal.name}={self._format_numeric_value(value)}")
            if y_value is None:
                y_value = value

        if y_value is not None and np.isfinite(y_value):
            self.h_line.setPos(y_value)
            self.h_line.show()
        else:
            self.h_line.hide()

        overlay_rows: list[tuple[str, float, str]] = []
        for order, signal_id in enumerate(self.selected_signal_ids[:6]):
            value = float(self._get_display_series(signal_id)[index])
            signal = self.signal_lookup[signal_id]
            color = self._series_color(order)
            overlay_rows.append((self._xt_display_signal_label(signal), value, color))

        self._set_overlay_text(self._format_overlay_rows(overlay_rows))
        self.v_line.show()
        self.plot_widget.viewport().update()
        self.emit_status(self._measurement_status or " | ".join(preview_parts))

    def _update_xy_cursor(self, x_value: float, y_value: float) -> None:
        if len(self.selected_signal_ids) != 2:
            self._hide_cursor()
            return

        assert self.parsed_log is not None
        if self._visible_indices.size == 0:
            self._hide_cursor(broadcast=True)
            return

        x_series = self._get_display_series(self.selected_signal_ids[0])[self._visible_indices]
        y_series = self._get_display_series(self.selected_signal_ids[1])[self._visible_indices]
        local_index = self._nearest_xy_local_index(x_series, y_series, x_value, y_value)
        if local_index is None:
            self._hide_cursor(broadcast=True)
            return
        index = int(self._visible_indices[local_index])
        self._current_cursor_index = index
        self._show_xy_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)

    def _show_xy_cursor_at_index(self, index: int) -> None:
        assert self.parsed_log is not None
        x_series = self._get_display_series(self.selected_signal_ids[0])
        y_series = self._get_display_series(self.selected_signal_ids[1])
        nearest_x = float(x_series[index])
        nearest_y = float(y_series[index])

        self.v_line.setPos(nearest_x)
        self.h_line.setPos(nearest_y)

        x_signal = self.signal_lookup[self.selected_signal_ids[0]]
        y_signal = self.signal_lookup[self.selected_signal_ids[1]]
        self._set_overlay_text(
            self._format_overlay_rows(
                [
                    (f"X - {self._display_signal_label(x_signal)}", nearest_x, self._series_color(0)),
                    (f"Y - {self._display_signal_label(y_signal)}", nearest_y, self._series_color(1)),
                ]
            )
        )
        self.v_line.show()
        self.h_line.show()
        self.plot_widget.viewport().update()
        status = f"Point {index} | {x_signal.name}={nearest_x:.6f} | {y_signal.name}={nearest_y:.6f}"
        self.emit_status(self._measurement_status or status)


class Plot3DPanel(BasePlotPanel):
    def __init__(
        self,
        activate_callback: Callable[[BasePlotPanel], None],
        remove_callback: Callable[[BasePlotPanel], None],
    ) -> None:
        super().__init__(mode="xyz", activate_callback=activate_callback, remove_callback=remove_callback)
        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.line_item = None
        self._default_distance = 40.0

        if GL_AVAILABLE:
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setMinimumHeight(280)
            self.grid_item = gl.GLGridItem()
            self.gl_widget.addItem(self.grid_item)
            try:
                self.axis_item = gl.GLAxisItem()
                self.gl_widget.addItem(self.axis_item)
            except Exception:  # pragma: no cover - depends on backend support
                self.axis_item = None
            self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
            self.gl_widget.setCameraPosition(distance=self._default_distance)
            self.content_layout.addWidget(self.gl_widget, 1)
        else:
            self.gl_widget = None
            self.axis_item = None
            missing_label = QLabel(
                "3D dependencies are missing, so the XYZ plot is unavailable.\n"
                f"Install PyOpenGL and restart the app.\nError: {GL_IMPORT_ERROR}"
            )
            missing_label.setWordWrap(True)
            missing_label.setStyleSheet("color: #b54708;")
            self.content_layout.addWidget(missing_label)

    @property
    def selection_limit(self) -> int | None:
        return 3

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        self.parsed_log = parsed_log
        self.signal_lookup = signal_lookup
        self.detail_label.setText("3D trajectory")

        if not GL_AVAILABLE or self.gl_widget is None:
            self.message_label.setText("XYZ plot unavailable. Install PyOpenGL first.")
            return

        if self.line_item is not None:
            self.gl_widget.removeItem(self.line_item)
            self.line_item = None

        if parsed_log is None:
            self.message_label.setText("No data loaded")
            return

        if len(self.selected_signal_ids) != 3:
            self.message_label.setText("Select exactly 3 signals for the XYZ plot in X, Y, Z order.")
            return

        indices = self._time_indices(parsed_log)
        if indices.size == 0:
            self.message_label.setText("No samples in the selected time range.")
            return

        x_signal_id, y_signal_id, z_signal_id = self.selected_signal_ids
        x_signal = signal_lookup[x_signal_id]
        y_signal = signal_lookup[y_signal_id]
        z_signal = signal_lookup[z_signal_id]
        points = np.column_stack(
            [
                parsed_log.get_series(x_signal_id)[indices],
                parsed_log.get_series(y_signal_id)[indices],
                parsed_log.get_series(z_signal_id)[indices],
            ]
        ).astype(np.float32)
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] == 0:
            self.message_label.setText("The selected XYZ signals contain no finite sample triples.")
            return
        center = points.mean(axis=0)
        centered_points = points - center

        self.line_item = gl.GLLinePlotItem(
            pos=centered_points,
            color=(0.12, 0.47, 0.71, 1.0),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        self.gl_widget.addItem(self.line_item)

        span = np.ptp(centered_points, axis=0)
        max_span = max(float(np.max(span)), 1.0)
        distance = max(max_span * 2.5, 20.0)
        grid_scale = max(max_span / 10.0, 1.0)
        self.grid_item.resetTransform()
        self.grid_item.scale(grid_scale, grid_scale, 1.0)
        if self.axis_item is not None:
            try:
                self.axis_item.setSize(max_span, max_span, max_span)
            except Exception:  # pragma: no cover - backend dependent
                pass
        self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.gl_widget.setCameraPosition(distance=distance)
        self._default_distance = distance
        self.message_label.setText(f"XYZ plot: X={x_signal.name} | Y={y_signal.name} | Z={z_signal.name}")
        self.emit_status(
            f"XYZ trajectory: X={x_signal.full_path} | Y={y_signal.full_path} | Z={z_signal.full_path}"
        )

    def reset_view(self) -> None:
        if self.gl_widget is not None:
            self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
            self.gl_widget.setCameraPosition(distance=self._default_distance)


class RobotPosePanel(BasePlotPanel):
    def __init__(
        self,
        activate_callback: Callable[[BasePlotPanel], None],
        remove_callback: Callable[[BasePlotPanel], None],
    ) -> None:
        super().__init__(mode="robot", activate_callback=activate_callback, remove_callback=remove_callback)
        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.robot_model: RobotModel | None = None
        self.robot_model_path: Path | None = None
        self.joint_signal_map: dict[str, str] = {}
        self.current_sample_index: int | None = None
        self.angle_unit = "deg"
        self._default_distance = 40.0
        self._last_center = np.zeros(3, dtype=float)
        self._view_center = np.zeros(3, dtype=float)
        self._auto_fit_pending = True
        self.line_item = None
        self.joint_item = None
        self.mesh_items: list[object] = []
        self._mesh_cache: dict[Path, LoadedMesh] = {}
        self._mesh_load_errors: dict[str, str] = {}
        self.world_axis_items: list[object] = []
        self.tool_axis_items: list[object] = []
        self._gl_init_requested = False
        self._gl_ready = False
        self._pending_sample_index: int | None = None
        self._pose_update_timer = QTimer(self)
        self._pose_update_timer.setSingleShot(True)
        self._pose_update_timer.setInterval(33)
        self._pose_update_timer.timeout.connect(self._apply_pending_sample_index)

        self.load_model_button = QPushButton("Load URDF/Xacro")
        self.load_model_button.clicked.connect(self._open_model_dialog)
        self.unit_toggle_button = QPushButton("Unit: deg")
        self.unit_toggle_button.clicked.connect(self._toggle_angle_unit)
        self.header_layout.insertWidget(2, self.load_model_button)
        self.header_layout.insertWidget(3, self.unit_toggle_button)

        self.summary_label = QLabel("Follow: waiting for a hovered/clicked sample in X-T")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #4b5563; padding-bottom: 4px;")
        self.content_layout.addWidget(self.summary_label)

        if GL_AVAILABLE:
            self.gl_widget = None
            self.grid_item = None
            self.axis_item = None
            self.gl_placeholder = QLabel("Initializing 3D robot view...")
            self.gl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.gl_placeholder.setMinimumHeight(320)
            self.gl_placeholder.setStyleSheet("color: #6b7280; background: #f8fafc; border: 1px solid #e5e7eb;")
            self.content_layout.addWidget(self.gl_placeholder, 1)
        else:
            self.gl_widget = None
            self.grid_item = None
            self.axis_item = None
            self.gl_placeholder = None
            missing_label = QLabel(
                "3D dependencies are missing, so the robot pose view is unavailable.\n"
                f"Install PyOpenGL and restart the app.\nError: {GL_IMPORT_ERROR}"
            )
            missing_label.setWordWrap(True)
            missing_label.setStyleSheet("color: #b54708;")
            self.content_layout.addWidget(missing_label)

        self._update_detail_label()
        self.message_label.setText("Load a URDF/Xacro model to preview the robot pose.")

    @property
    def supports_time_range(self) -> bool:
        return False

    def _update_detail_label(self) -> None:
        model_name = self.robot_model_path.name if self.robot_model_path is not None else "No model"
        self.detail_label.setText(f"Robot Pose | {model_name} | Angle unit={self.angle_unit}")

    def _toggle_angle_unit(self) -> None:
        self.angle_unit = "rad" if self.angle_unit == "deg" else "deg"
        self.unit_toggle_button.setText(f"Unit: {self.angle_unit}")
        self._update_detail_label()
        self._refresh_pose()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        if GL_AVAILABLE and not self._gl_init_requested:
            self._gl_init_requested = True
            QTimer.singleShot(0, self._ensure_gl_widget)

    def _ensure_gl_widget(self) -> None:
        if not GL_AVAILABLE or self._gl_ready:
            return

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMinimumHeight(320)
        try:
            self.gl_widget.setBackgroundColor("#f7fafc")
        except Exception:  # pragma: no cover - backend dependent
            pass
        self.grid_item = gl.GLGridItem()
        self.gl_widget.addItem(self.grid_item)
        try:
            self.grid_item.setColor((80, 80, 80, 90))
        except Exception:  # pragma: no cover - backend dependent
            pass
        try:
            self.axis_item = gl.GLAxisItem()
            self.gl_widget.addItem(self.axis_item)
        except Exception:  # pragma: no cover - backend dependent
            self.axis_item = None
        self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.gl_widget.setCameraPosition(distance=self._default_distance)

        if self.gl_placeholder is not None:
            self.content_layout.replaceWidget(self.gl_placeholder, self.gl_widget)
            self.gl_placeholder.hide()
            self.gl_placeholder.deleteLater()
            self.gl_placeholder = None
        else:
            self.content_layout.addWidget(self.gl_widget, 1)

        self._gl_ready = True
        self._refresh_pose()

    def _open_model_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Robot Model",
            str(self.robot_model_path.parent) if self.robot_model_path is not None else "",
            "Robot Model (*.urdf *.xacro);;URDF Files (*.urdf);;Xacro Files (*.xacro);;All Files (*)",
        )
        if filename:
            self.load_model_from_path(Path(filename))

    def load_model_from_path(self, path: Path) -> None:
        try:
            self.robot_model = load_robot_model(path)
        except RobotModelError as exc:
            self.message_label.setText(str(exc))
            self.emit_status(str(exc))
            return

        self.robot_model_path = path.expanduser().resolve()
        self.joint_signal_map.clear()
        self._mesh_cache.clear()
        self._mesh_load_errors.clear()
        self._auto_fit_pending = True
        self._update_detail_label()
        self._auto_map_joint_signals()
        self._refresh_pose()

    @staticmethod
    def _joint_index_from_text(text: str) -> int | None:
        match = re.search(r"(?:^|[^a-z])j(?:oint)?[_-]?(\d+)$", text.strip().lower())
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _joint_sort_key(cls, signal: SignalNode) -> tuple[int, int, str]:
        text_parts = [signal.name, signal.full_path, *signal.path_parts]
        has_joint_token = any("joint" in part.lower() for part in text_parts)
        joint_index = None
        for part in text_parts:
            joint_index = cls._joint_index_from_text(part)
            if joint_index is not None:
                break
        return (0 if has_joint_token else 1, joint_index if joint_index is not None else 10**6, signal.full_path)

    @staticmethod
    def _path_score(signal: SignalNode) -> tuple[int, int, int, str]:
        parts_lower = [part.lower() for part in signal.path_parts]
        full_path_lower = signal.full_path.lower()

        source_score = 100
        if "joint" in parts_lower:
            source_score = 0
        elif any(part.endswith("mcjoint") for part in parts_lower):
            source_score = 10
        elif any("joint" in part for part in parts_lower):
            source_score = 20

        metric_score = 50
        if "pos" in parts_lower:
            metric_score = 0
        elif any(part.endswith("jpos") for part in parts_lower):
            metric_score = 5
        elif any("pos" in part for part in parts_lower):
            metric_score = 10

        penalty = 0
        bad_keywords = (
            "vel",
            "acc",
            "torq",
            "ctrwd",
            "stswd",
            "modefb",
            "actpls",
            "setpls",
            "followerrs",
            "ipjvel",
            "damp",
            "outpos",
            "status",
            "port",
        )
        for keyword in bad_keywords:
            if keyword in full_path_lower:
                penalty += 20
        if "**" in full_path_lower and "mcjoint" not in full_path_lower:
            penalty += 10

        return (source_score, metric_score, penalty, signal.full_path)

    @classmethod
    def _looks_like_joint_signal(cls, signal: SignalNode) -> bool:
        text_parts = [signal.name, signal.full_path, *signal.path_parts]
        if any("joint" in part.lower() for part in text_parts):
            return True
        return any(cls._joint_index_from_text(part) is not None for part in text_parts)

    @classmethod
    def _looks_like_joint_position_signal(cls, signal: SignalNode) -> bool:
        source_score, metric_score, penalty, _ = cls._path_score(signal)
        return (
            cls._looks_like_joint_signal(signal)
            and source_score <= 20
            and metric_score <= 10
            and penalty == 0
        )

    def _auto_map_joint_signals(self) -> None:
        if self.robot_model is None or self.parsed_log is None:
            self.joint_signal_map.clear()
            return

        self.joint_signal_map = self._build_joint_signal_map(self.robot_model, self.parsed_log)

    @classmethod
    def _build_joint_signal_map(
        cls,
        robot_model: RobotModel,
        parsed_log: ParsedLog,
    ) -> dict[str, str]:
        joint_signal_map: dict[str, str] = {}

        candidates = [
            signal
            for signal in parsed_log.signals
            if signal.available
            and signal.signal_id in parsed_log.signals_by_id
            and cls._looks_like_joint_position_signal(signal)
        ]
        candidates.sort(key=lambda signal: (*cls._path_score(signal), *cls._joint_sort_key(signal)))
        movable_joints = robot_model.movable_joints
        if not candidates or not movable_joints:
            return joint_signal_map

        candidate_by_index: dict[int, SignalNode] = {}
        for signal in candidates:
            for part in (signal.name, signal.full_path, *signal.path_parts):
                joint_index = cls._joint_index_from_text(part)
                if joint_index is None:
                    continue
                previous = candidate_by_index.get(joint_index)
                if previous is None or cls._path_score(signal) < cls._path_score(previous):
                    candidate_by_index[joint_index] = signal
                    break

        for joint in movable_joints:
            joint_index = cls._joint_index_from_text(joint.name)
            if joint_index is not None and joint_index in candidate_by_index:
                signal = candidate_by_index[joint_index]
                joint_signal_map[joint.name] = signal.signal_id
        return joint_signal_map

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        previous_log = self.parsed_log
        self.parsed_log = parsed_log
        self.signal_lookup = signal_lookup
        if parsed_log is not previous_log:
            self._auto_fit_pending = True
            self._pose_update_timer.stop()
            self._pending_sample_index = None
            self.current_sample_index = 0 if parsed_log is not None and parsed_log.time_seconds.size else None
        if parsed_log is not None:
            sample_count = parsed_log.time_seconds.shape[0]
            if sample_count > 0 and (
                self.current_sample_index is None
                or self.current_sample_index < 0
                or self.current_sample_index >= sample_count
            ):
                self.current_sample_index = 0

        self._auto_map_joint_signals()
        self._refresh_pose()

    def sync_sample_index(self, index: int | None) -> None:
        if index is None or self.parsed_log is None:
            return
        sample_count = self.parsed_log.time_seconds.shape[0]
        if sample_count == 0:
            return
        clamped_index = max(0, min(int(index), sample_count - 1))
        if clamped_index == self.current_sample_index and self._pending_sample_index is None:
            return
        self._pending_sample_index = clamped_index
        if not self._pose_update_timer.isActive():
            self._pose_update_timer.start()

    def _apply_pending_sample_index(self) -> None:
        if self._pending_sample_index is None:
            return
        self.current_sample_index = self._pending_sample_index
        self._pending_sample_index = None
        self._refresh_pose()

    def focus_sample_index(self, index: int) -> None:
        if self.parsed_log is None or self.parsed_log.time_seconds.size == 0:
            return
        self._pose_update_timer.stop()
        self._pending_sample_index = None
        sample_count = self.parsed_log.time_seconds.shape[0]
        self.current_sample_index = max(0, min(int(index), sample_count - 1))
        self._refresh_pose()

    def _joint_values_for_index(self, sample_index: int) -> dict[str, float]:
        if self.parsed_log is None or self.robot_model is None:
            return {}

        joint_values: dict[str, float] = {}
        for joint in self.robot_model.movable_joints:
            signal_id = self.joint_signal_map.get(joint.name)
            if signal_id is None or signal_id not in self.parsed_log.signals_by_id:
                continue
            raw_value = float(self.parsed_log.get_series(signal_id)[sample_index])
            if not np.isfinite(raw_value):
                continue
            if joint.joint_type in {"revolute", "continuous"} and self.angle_unit == "deg":
                raw_value = float(np.deg2rad(raw_value))
            joint_values[joint.name] = raw_value
        return joint_values

    def _format_joint_summary(self, sample_index: int) -> str:
        if self.parsed_log is None or self.robot_model is None:
            return "Follow: unavailable"

        joint_parts: list[str] = []
        for joint in self.robot_model.movable_joints[:6]:
            signal_id = self.joint_signal_map.get(joint.name)
            if signal_id is None or signal_id not in self.parsed_log.signals_by_id:
                continue
            raw_value = float(self.parsed_log.get_series(signal_id)[sample_index])
            signal = self.signal_lookup.get(signal_id)
            source_label = signal.full_path if signal is not None else signal_id
            formatted_value = f"{raw_value:.3f}" if np.isfinite(raw_value) else "n/a"
            joint_parts.append(f"{joint.name}<-{source_label}={formatted_value}")

        if not joint_parts:
            return f"Follow: sample={sample_index} | no mapped joint values"

        time_seconds = float(self.parsed_log.time_seconds[sample_index])
        return f"Follow: sample={sample_index} | t={time_seconds:.3f}s | " + " | ".join(joint_parts)

    def _clear_robot_item(self) -> None:
        if self.gl_widget is not None and self.line_item is not None:
            self.gl_widget.removeItem(self.line_item)
            self.line_item = None
        if self.gl_widget is not None and self.joint_item is not None:
            self.gl_widget.removeItem(self.joint_item)
            self.joint_item = None
        if self.gl_widget is not None:
            for item in self.mesh_items:
                self.gl_widget.removeItem(item)
            for item in self.world_axis_items:
                self.gl_widget.removeItem(item)
            for item in self.tool_axis_items:
                self.gl_widget.removeItem(item)
        self.mesh_items.clear()
        self.world_axis_items.clear()
        self.tool_axis_items.clear()

    @staticmethod
    def _build_axis_items(
        origin: np.ndarray,
        rotation: np.ndarray,
        length: float,
        colors: tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
        *,
        width: float,
    ) -> list[object]:
        axis_specs = (
            (rotation[:, 0], colors[0]),
            (rotation[:, 1], colors[1]),
            (rotation[:, 2], colors[2]),
        )
        items: list[object] = []
        for axis_vector, color in axis_specs:
            points = np.vstack([origin, origin + axis_vector * length]).astype(np.float32)
            items.append(
                gl.GLLinePlotItem(
                    pos=points,
                    color=color,
                    width=width,
                    antialias=True,
                    mode="line_strip",
                )
            )
        return items

    def _load_visual_mesh(self, mesh_path: Path) -> LoadedMesh | None:
        cached_mesh = self._mesh_cache.get(mesh_path)
        if cached_mesh is not None:
            return cached_mesh

        try:
            loaded_mesh = load_mesh(mesh_path)
        except MeshLoadError as exc:
            self._mesh_load_errors[str(mesh_path)] = str(exc)
            return None

        self._mesh_cache[mesh_path] = loaded_mesh
        return loaded_mesh

    @staticmethod
    def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homogeneous = np.ones((vertices.shape[0], 4), dtype=np.float32)
        homogeneous[:, :3] = vertices.astype(np.float32, copy=False)
        return (homogeneous @ transform.T.astype(np.float32))[:, :3]

    def _refresh_pose(self) -> None:
        self._update_detail_label()
        if not GL_AVAILABLE:
            self.message_label.setText("Robot pose view unavailable. Install PyOpenGL first.")
            self.summary_label.setText("Follow: unavailable")
            return
        if self.gl_widget is None:
            self.message_label.setText("Preparing 3D robot view...")
            self.summary_label.setText("Follow: initializing 3D view")
            return

        self._clear_robot_item()

        if self.robot_model is None:
            self.message_label.setText("Load a URDF/Xacro model to preview the robot pose.")
            self.summary_label.setText("Follow: waiting for robot model")
            return

        sample_index: int | None = None
        time_seconds: float | None = None
        joint_values: dict[str, float] = {}
        follow_summary = "Follow: default robot pose"
        if self.parsed_log is None:
            follow_summary = "Follow: default robot pose | no log loaded"
        elif not self.robot_model.movable_joints:
            follow_summary = "Follow: default robot pose | robot model has no movable joints"
        elif not self.joint_signal_map:
            follow_summary = "Follow: default robot pose | no mapped joint signals"
        else:
            sample_count = self.parsed_log.time_seconds.shape[0]
            if sample_count > 0:
                sample_index = (
                    0 if self.current_sample_index is None else max(0, min(self.current_sample_index, sample_count - 1))
                )
                time_seconds = float(self.parsed_log.time_seconds[sample_index])
                joint_values = self._joint_values_for_index(sample_index)
                follow_summary = self._format_joint_summary(sample_index)
            else:
                follow_summary = "Follow: default robot pose | log contains no valid samples"

        state = self.robot_model.compute_state(joint_values)
        segments = state.segments
        if not segments:
            self.message_label.setText("The robot model produced no drawable segments.")
            return

        line_points = np.array([point for segment in segments for point in segment], dtype=np.float32)
        mesh_points: list[np.ndarray] = []
        mesh_drawn_count = 0
        mesh_skipped_count = 0
        for visual_mesh in self.robot_model.visual_meshes:
            if visual_mesh.path is None:
                self._mesh_load_errors[visual_mesh.filename] = f"无法根据 URDF 路径解析 mesh: {visual_mesh.filename}"
                mesh_skipped_count += 1
                continue

            loaded_mesh = self._load_visual_mesh(visual_mesh.path)
            if loaded_mesh is None:
                mesh_skipped_count += 1
                continue
            link_transform = state.link_transforms.get(visual_mesh.link_name)
            if link_transform is None:
                mesh_skipped_count += 1
                continue

            scaled_vertices = loaded_mesh.vertices * visual_mesh.scale.astype(np.float32)
            transformed_vertices = self._transform_vertices(
                scaled_vertices,
                link_transform @ visual_mesh.origin_transform,
            )
            mesh_points.append(transformed_vertices)
            mesh_data = gl.MeshData(
                vertexes=transformed_vertices.astype(np.float32, copy=False),
                faces=loaded_mesh.faces,
            )
            mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                color=ROBOT_MESH_COLOR,
                smooth=False,
                drawEdges=True,
                edgeColor=ROBOT_MESH_EDGE_COLOR,
                shader="shaded",
                glOptions="translucent",
            )
            self.mesh_items.append(mesh_item)
            self.gl_widget.addItem(mesh_item)
            mesh_drawn_count += 1

        if mesh_points:
            all_points = np.vstack([line_points.reshape(-1, 3), *mesh_points])
        else:
            all_points = line_points.reshape(-1, 3)
        min_point = all_points.min(axis=0)
        max_point = all_points.max(axis=0)
        center = (min_point + max_point) / 2.0
        self._last_center = center.astype(float)

        self.line_item = gl.GLLinePlotItem(
            pos=line_points.reshape(-1, 3).astype(np.float32),
            color=ROBOT_LINK_COLOR,
            width=4,
            antialias=True,
            mode="lines",
        )
        self.gl_widget.addItem(self.line_item)

        if state.joint_positions:
            joint_points = np.array(state.joint_positions, dtype=np.float32)
            self.joint_item = gl.GLScatterPlotItem(
                pos=joint_points,
                color=ROBOT_JOINT_COLOR,
                size=10.0,
                pxMode=True,
            )
            self.gl_widget.addItem(self.joint_item)

        actual_span = max(float(np.max(max_point - min_point)), 1e-6)
        grid_scale = max(actual_span / 6.0, 0.05)
        distance = max(actual_span * 2.4, 1.2)
        axis_length = max(actual_span * 0.22, 0.08)
        self.grid_item.resetTransform()
        self.grid_item.scale(grid_scale, grid_scale, 1.0)
        if self.axis_item is not None:
            try:
                self.axis_item.setSize(axis_length * 0.6, axis_length * 0.6, axis_length * 0.6)
            except Exception:  # pragma: no cover - backend dependent
                pass
        self.world_axis_items = self._build_axis_items(
            origin=np.zeros(3, dtype=float),
            rotation=np.eye(3, dtype=float),
            length=axis_length,
            colors=(WORLD_AXIS_X_COLOR, WORLD_AXIS_Y_COLOR, WORLD_AXIS_Z_COLOR),
            width=2,
        )
        for item in self.world_axis_items:
            self.gl_widget.addItem(item)

        tool_transform = state.tool_transform
        self.tool_axis_items = self._build_axis_items(
            origin=tool_transform[:3, 3],
            rotation=tool_transform[:3, :3],
            length=max(axis_length * 0.6, 0.05),
            colors=(AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR),
            width=3,
        )
        for item in self.tool_axis_items:
            self.gl_widget.addItem(item)

        if self._auto_fit_pending:
            self._view_center = center.astype(float)
            self._default_distance = distance
            self.gl_widget.opts["center"] = QVector3D(
                float(self._view_center[0]),
                float(self._view_center[1]),
                float(self._view_center[2]),
            )
            self.gl_widget.setCameraPosition(distance=self._default_distance)
            self._auto_fit_pending = False

        mapped_count = len(joint_values)
        total_count = len(self.robot_model.movable_joints)
        mesh_status = ""
        if self.robot_model.visual_meshes:
            mesh_status = f" | meshes={mesh_drawn_count}/{len(self.robot_model.visual_meshes)}"
            if mesh_skipped_count:
                mesh_status += f" skipped={mesh_skipped_count}"
                if self._mesh_load_errors:
                    first_error = next(iter(self._mesh_load_errors.values()))
                    mesh_status += f" | mesh error={first_error[:120]}"
        pose_status = "default pose" if sample_index is None or time_seconds is None else (
            f"sample={sample_index} | t={time_seconds:.3f}s"
        )
        self.message_label.setText(
            f"Robot pose | {pose_status} | mapped joints={mapped_count}/{total_count}{mesh_status}"
        )
        self.summary_label.setText(follow_summary)
        self.emit_status(
            f"Robot pose | {pose_status} | model={self.robot_model.name}"
        )

    def reset_view(self) -> None:
        if self.gl_widget is not None:
            self._view_center = self._last_center.copy()
            self.gl_widget.opts["center"] = QVector3D(
                float(self._view_center[0]),
                float(self._view_center[1]),
                float(self._view_center[2]),
            )
            self.gl_widget.setCameraPosition(distance=self._default_distance)


class MainPalette:
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]
