from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QHeaderView,
    QPushButton,
    QProgressDialog,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .events import EventDetectionSummary, detect_events_result
from .models import LogEvent, ParsedLog, SignalNode
from .parser import ParseError, parse_log_file
from .plot_panels import BasePlotPanel, Plot2DPanel, Plot3DPanel, RobotPosePanel


class PlotSubWindow(QMdiSubWindow):
    def __init__(self, panel: BasePlotPanel, close_callback: Callable[[BasePlotPanel], None]) -> None:
        super().__init__()
        self._panel = panel
        self._close_callback = close_callback
        self._managed_close = False
        self.setWidget(panel)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)
        self.setWindowFlags(
            Qt.WindowType.SubWindow
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._managed_close:
            super().closeEvent(event)
            return
        event.ignore()
        self._close_callback(self._panel)

    def close_from_manager(self) -> None:
        self._managed_close = True
        self.close()


class MainWindow(QMainWindow):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("机器人状态曲线查看器")
        self.resize(1580, 920)
        self.setAcceptDrops(True)

        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.signal_item_map: dict[str, QTreeWidgetItem] = {}
        self.panels: list[BasePlotPanel] = []
        self.panel_windows: dict[BasePlotPanel, PlotSubWindow] = {}
        self._drop_targets: list[QWidget] = []
        self.active_panel: BasePlotPanel | None = None
        self.detected_events: list[LogEvent] = []
        self.current_sample_index: int | None = None
        self._suppress_item_changed = False
        self._initial_path = initial_path

        self._build_ui()
        self.add_panel("xt")

        if self._initial_path is not None:
            self.load_file(self._initial_path)

    @staticmethod
    def _extract_dropped_log_path(event) -> Path | None:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None

        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() == ".txt":
                return path
        return None

    def _register_drop_target(self, widget: QWidget | None) -> None:
        if widget is None or widget in self._drop_targets:
            return
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)
        self._drop_targets.append(widget)

    def _unregister_drop_targets(self, *roots: QWidget) -> None:
        retained_targets: list[QWidget] = []
        for widget in self._drop_targets:
            owned_by_panel = any(widget is root or root.isAncestorOf(widget) for root in roots)
            if owned_by_panel:
                widget.removeEventFilter(self)
            else:
                retained_targets.append(widget)
        self._drop_targets = retained_targets

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt naming
        if self._extract_dropped_log_path(event) is not None:
            event.acceptProposedAction()
            self.cursor_label.setText("松开鼠标以打开日志文件")
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 - Qt naming
        if self._extract_dropped_log_path(event) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt naming
        path = self._extract_dropped_log_path(event)
        if path is None:
            event.ignore()
            self.cursor_label.setText("仅支持拖入本地 txt 日志文件")
            return

        event.acceptProposedAction()
        self.load_file(path)

    def eventFilter(self, watched, event) -> bool:
        if watched in self._drop_targets:
            if event.type() == QEvent.Type.DragEnter:
                self.dragEnterEvent(event)
                return event.isAccepted()
            if event.type() == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return event.isAccepted()
            if event.type() == QEvent.Type.Drop:
                self.dropEvent(event)
                return event.isAccepted()
        return super().eventFilter(watched, event)

    def _build_ui(self) -> None:
        open_action = QAction("打开文件", self)
        open_action.triggered.connect(self.open_file_dialog)
        self.menuBar().addAction(open_action)

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        controls_layout = QHBoxLayout()
        self.open_button = QPushButton("打开文件")
        self.clear_button = QPushButton("清空选择")
        self.reset_button = QPushButton("重置视图")
        self.add_xt_button = QPushButton("新增 X-T 图")
        self.add_xy_button = QPushButton("新增 X-Y 图")
        self.add_xyz_button = QPushButton("新增 XYZ 图")
        self.remove_panel_button = QPushButton("关闭当前图")
        self.add_robot_button = QPushButton("Robot")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索字段，例如 J1 / Tcp / ErrCode")

        range_layout = QHBoxLayout()
        self.time_start_input = QDoubleSpinBox()
        self.time_end_input = QDoubleSpinBox()
        for spin_box in (self.time_start_input, self.time_end_input):
            spin_box.setDecimals(3)
            spin_box.setRange(0.0, 0.0)
            spin_box.setSingleStep(0.100)
            spin_box.setSuffix(" s")
            spin_box.setEnabled(False)
        self.apply_time_range_button = QPushButton("应用时间区间")
        self.clear_time_range_button = QPushButton("全部时间")
        self.apply_time_range_button.setEnabled(False)
        self.clear_time_range_button.setEnabled(False)

        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.add_xt_button)
        controls_layout.addWidget(self.add_xy_button)
        controls_layout.addWidget(self.add_xyz_button)
        controls_layout.addWidget(self.add_robot_button)
        controls_layout.addWidget(self.remove_panel_button)
        controls_layout.addWidget(self.search_input, 1)

        range_layout.addWidget(QLabel("时间区间:"))
        range_layout.addWidget(QLabel("Start"))
        range_layout.addWidget(self.time_start_input)
        range_layout.addWidget(QLabel("End"))
        range_layout.addWidget(self.time_end_input)
        range_layout.addWidget(self.apply_time_range_button)
        range_layout.addWidget(self.clear_time_range_button)
        range_layout.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("信号字段")
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setAllColumnsShowFocus(False)
        self.tree.itemChanged.connect(self._on_tree_item_changed)

        self.mdi_area = QMdiArea()
        self.mdi_area.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.mdi_area.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder)
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)

        splitter.addWidget(self.tree)
        splitter.addWidget(self.mdi_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1220])

        self.event_table = QTableWidget(0, 5)
        self.event_table.setHorizontalHeaderLabels(["时间(s)", "原始时间", "字段", "变化", "类型"])
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.event_table.setSortingEnabled(False)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.event_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        event_bar_layout = QHBoxLayout()
        self.toggle_events_button = QPushButton("隐藏事件")
        self.event_summary_label = QLabel("事件: 未加载")
        self.event_summary_label.setStyleSheet("color: #666666;")
        event_bar_layout.addWidget(self.toggle_events_button)
        event_bar_layout.addWidget(self.event_summary_label, 1)

        event_panel = QWidget()
        event_panel_layout = QVBoxLayout(event_panel)
        event_panel_layout.setContentsMargins(0, 0, 0, 0)
        event_panel_layout.setSpacing(4)
        event_panel_layout.addLayout(event_bar_layout)
        event_panel_layout.addWidget(self.event_table)
        self.event_panel = event_panel

        workspace_splitter = QSplitter(Qt.Orientation.Vertical)
        workspace_splitter.addWidget(splitter)
        workspace_splitter.addWidget(self.event_panel)
        workspace_splitter.setStretchFactor(0, 1)
        workspace_splitter.setStretchFactor(1, 0)
        workspace_splitter.setSizes([720, 170])
        self.workspace_splitter = workspace_splitter
        self._event_panel_expanded_height = 170
        self.event_table.hide()
        self._register_drop_target(central)
        self._register_drop_target(self.tree)
        self._register_drop_target(self.mdi_area)
        self._register_drop_target(self.mdi_area.viewport())
        self._register_drop_target(self.event_panel)
        self._register_drop_target(self.event_table)
        self.toggle_events_button.setText("显示事件")

        root_layout.addLayout(controls_layout)
        root_layout.addLayout(range_layout)
        root_layout.addWidget(workspace_splitter, 1)
        self.setCentralWidget(central)

        self.info_label = QLabel("未加载文件")
        self.cursor_label = QLabel("未选择曲线")
        self.statusBar().addWidget(self.info_label, 1)
        self.statusBar().addPermanentWidget(self.cursor_label, 1)

        self.open_button.clicked.connect(self.open_file_dialog)
        self.clear_button.clicked.connect(self.clear_selection)
        self.reset_button.clicked.connect(self.reset_view)
        self.add_xt_button.clicked.connect(lambda: self.add_panel("xt"))
        self.add_xy_button.clicked.connect(lambda: self.add_panel("xy"))
        self.add_xyz_button.clicked.connect(lambda: self.add_panel("xyz"))
        self.add_robot_button.clicked.connect(lambda: self.add_panel("robot"))
        self.remove_panel_button.clicked.connect(self.remove_active_panel)
        self.apply_time_range_button.clicked.connect(self.apply_time_range)
        self.clear_time_range_button.clicked.connect(self.clear_time_range)
        self.toggle_events_button.clicked.connect(self.toggle_event_panel)
        self.event_table.cellDoubleClicked.connect(self._on_event_row_activated)
        self.search_input.textChanged.connect(self.apply_filter)
        self._set_event_panel_collapsed()

    def open_file_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "打开机器人日志",
            "",
            "Log Files (*.txt);;All Files (*)",
        )
        if filename:
            self.load_file(Path(filename))

    def load_file(self, path: Path) -> None:
        progress = QProgressDialog("正在读取文件...", "取消", 0, 100, self)
        progress.setWindowTitle("加载日志")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def report_progress(percent: int, message: str) -> bool:
            progress.setValue(percent)
            progress.setLabelText(message)
            QApplication.processEvents()
            return not progress.wasCanceled()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            parsed = parse_log_file(path, progress_callback=report_progress)
            if progress.wasCanceled():
                return
            progress.setValue(99)
            progress.setLabelText("正在识别事件...")
            QApplication.processEvents()
            event_result = detect_events_result(parsed)
            detected_events = event_result.events
            progress.setValue(100)
        except (OSError, ParseError) as exc:
            if progress.wasCanceled():
                return
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            progress.close()

        self.parsed_log = parsed
        self.detected_events = detected_events
        self.current_sample_index = 0 if parsed.time_seconds.shape[0] else None
        self.signal_lookup = {signal.signal_id: signal for signal in parsed.signals}
        self._populate_tree(parsed.signals)
        self._populate_event_table(detected_events, event_result.summary)
        max_time_seconds = float(parsed.time_seconds[-1])
        for panel in self.panels:
            panel.selected_signal_ids = [
                signal_id for signal_id in panel.selected_signal_ids if signal_id in self.signal_lookup
            ]
            panel.clamp_time_range(max_time_seconds)
            panel.update_plot(self.parsed_log, self.signal_lookup)
            panel.sync_sample_index(self.current_sample_index)
        if self.active_panel is not None:
            self._sync_tree_from_active_panel()
            self._sync_time_controls_from_active_panel()

        self.info_label.setText(
            "文件: "
            f"{parsed.meta.path.name} | 采样点: {parsed.meta.sample_count} | 字段: {parsed.meta.field_count} | "
            f"起始: {parsed.meta.start_time_raw} | 结束: {parsed.meta.end_time_raw} | "
            f"跳过坏行: {parsed.meta.skipped_rows}"
        )
        self.cursor_label.setText("文件已加载，可在左侧勾选信号开始绘图")
        self.setWindowTitle(f"机器人状态曲线查看器 - {parsed.meta.path.name}")
        self.reset_view()

        if parsed.meta.skipped_rows:
            QMessageBox.warning(
                self,
                "部分数据已跳过",
                f"读取过程中跳过了 {parsed.meta.skipped_rows} 行异常数据，其余有效数据仍可正常查看。",
            )

    def _populate_tree(self, signals: list[SignalNode]) -> None:
        self._suppress_item_changed = True
        self.tree.clear()
        self.signal_item_map.clear()

        created_items: dict[tuple[str, ...], QTreeWidgetItem] = {}

        for signal in signals:
            parent = self.tree.invisibleRootItem()
            current_path: list[str] = []

            for depth, part in enumerate(signal.path_parts):
                current_path.append(part)
                key = tuple(current_path)
                item = created_items.get(key)

                if item is None:
                    item = QTreeWidgetItem([part])
                    item.setToolTip(0, " / ".join(current_path))
                    created_items[key] = item
                    parent.addChild(item)

                parent = item

                if depth == len(signal.path_parts) - 1:
                    item.setData(0, Qt.ItemDataRole.UserRole, signal.signal_id)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                    item.setToolTip(0, signal.full_path)
                    self.signal_item_map[signal.signal_id] = item
                    if not signal.available:
                        item.setDisabled(True)
                        item.setText(0, f"{part} (无数据列)")

        for index in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(index).setExpanded(True)

        self._suppress_item_changed = False
        self.apply_filter(self.search_input.text())
        self._sync_tree_from_active_panel()

    def _populate_event_table(
        self,
        events: list[LogEvent],
        summary: EventDetectionSummary | None = None,
    ) -> None:
        self.event_table.setRowCount(0)
        if not events:
            self.event_table.insertRow(0)
            self.event_table.setItem(0, 2, QTableWidgetItem("未识别到状态/报警事件字段"))
            self.event_table.setToolTip("当前文件未识别到状态/报警事件字段")
            self.event_summary_label.setText("事件: 0")
            return

        for row, event in enumerate(events):
            self.event_table.insertRow(row)
            values = [
                f"{event.time_seconds:.3f}",
                event.time_raw,
                event.signal_path,
                f"{event.previous_value:g} -> {event.current_value:g}",
                event.event_type,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(event.signal_path)
                item.setData(Qt.ItemDataRole.UserRole, event.sample_index)
                self.event_table.setItem(row, column, item)

        if summary is not None and summary.truncated:
            tooltip = (
                f"识别到 {summary.total_candidates} 个状态/报警事件，"
                f"当前显示 {summary.returned_count} 个，已省略 {summary.truncated_count} 个。"
            )
            label = f"事件: {summary.returned_count}/{summary.total_candidates} (已省略 {summary.truncated_count})"
        else:
            tooltip = f"识别到 {len(events)} 个状态/报警事件，双击可跳转到对应采样点"
            label = f"事件: {len(events)}"
        self.event_table.setToolTip(tooltip)
        self.event_summary_label.setText(label)

    def toggle_event_panel(self) -> None:
        visible = self.event_table.isVisible()
        sizes = self.workspace_splitter.sizes()
        if visible and len(sizes) >= 2:
            self._event_panel_expanded_height = max(sizes[1], self._event_panel_expanded_height)

        self.event_table.setVisible(not visible)
        self.toggle_events_button.setText("显示事件" if visible else "隐藏事件")
        QApplication.processEvents()

        if visible:
            self._set_event_panel_collapsed()
        else:
            total_height = sum(self.workspace_splitter.sizes())
            expanded_height = min(self._event_panel_expanded_height, max(total_height // 2, 1))
            self.workspace_splitter.setSizes([max(total_height - expanded_height, 1), expanded_height])

    def _set_event_panel_collapsed(self) -> None:
        total_height = sum(self.workspace_splitter.sizes())
        collapsed_height = self.toggle_events_button.sizeHint().height() + 10
        self.workspace_splitter.setSizes([max(total_height - collapsed_height, 1), collapsed_height])

    def add_panel(self, mode: str) -> None:
        freeze_updates = mode in {"xyz", "robot"}
        if freeze_updates:
            self.setUpdatesEnabled(False)
            self.mdi_area.setUpdatesEnabled(False)
            self.mdi_area.viewport().setUpdatesEnabled(False)

        try:
            if mode == "xyz":
                panel: BasePlotPanel = Plot3DPanel(self.set_active_panel, self.remove_panel)
            elif mode == "robot":
                panel = RobotPosePanel(self.set_active_panel, self.remove_panel)
            else:
                panel = Plot2DPanel(mode, self.set_active_panel, self.remove_panel)

            subwindow = PlotSubWindow(panel, self.remove_panel)
            panel.set_status_callback(lambda text, owner=panel: self._update_panel_status(owner, text))
            if isinstance(panel, Plot2DPanel):
                panel.set_cursor_sync_callback(lambda index, owner=panel: self._sync_plot_cursors(owner, index))

            self.mdi_area.addSubWindow(subwindow)
            self.panels.append(panel)
            self.panel_windows[panel] = subwindow
            self._register_drop_target(subwindow)
            self._register_drop_target(panel)
            if isinstance(panel, Plot2DPanel):
                self._register_drop_target(panel.plot_widget)
                self._register_drop_target(panel.plot_widget.viewport())
            elif isinstance(panel, Plot3DPanel):
                self._register_drop_target(panel.gl_widget)
            elif isinstance(panel, RobotPosePanel):
                self._register_drop_target(panel.gl_widget)
            self._update_panel_titles()
            panel.update_plot(self.parsed_log, self.signal_lookup)
            if self.current_sample_index is not None:
                panel.sync_sample_index(self.current_sample_index)
            self._position_new_panel(subwindow, mode)
            subwindow.show()
            self.set_active_panel(panel)
        finally:
            if freeze_updates:
                self.mdi_area.viewport().setUpdatesEnabled(True)
                self.mdi_area.setUpdatesEnabled(True)
                self.setUpdatesEnabled(True)
                self.mdi_area.viewport().update()
                self.mdi_area.update()

    def remove_panel(self, panel: BasePlotPanel) -> None:
        if panel not in self.panels:
            return

        was_active = self.active_panel is panel
        subwindow = self.panel_windows.pop(panel, None)
        self.panels.remove(panel)

        if subwindow is not None:
            self._unregister_drop_targets(subwindow, panel)
            subwindow.close_from_manager()
            self.mdi_area.removeSubWindow(subwindow)
            panel.setParent(None)
            subwindow.deleteLater()

        panel.deleteLater()

        if not self.panels:
            self.active_panel = None
            self.add_panel("xt")
            return

        self._update_panel_titles()
        if was_active:
            self.set_active_panel(self.panels[-1])

    def remove_active_panel(self) -> None:
        if self.active_panel is not None:
            self.remove_panel(self.active_panel)

    def _update_panel_titles(self) -> None:
        mode_names = {"xt": "X-T", "xy": "X-Y", "xyz": "XYZ", "robot": "Robot"}
        for index, panel in enumerate(self.panels, start=1):
            title = f"图 {index} - {mode_names.get(panel.mode, panel.mode.upper())}"
            panel.set_panel_title(title)
            subwindow = self.panel_windows.get(panel)
            if subwindow is not None:
                subwindow.setWindowTitle(title)

    def set_active_panel(self, panel: BasePlotPanel) -> None:
        if panel not in self.panels:
            return

        self.active_panel = panel
        for current in self.panels:
            current.set_active(current is panel)

        subwindow = self.panel_windows.get(panel)
        if subwindow is not None and self.mdi_area.activeSubWindow() is not subwindow:
            self.mdi_area.setActiveSubWindow(subwindow)

        self._sync_tree_from_active_panel()
        self._sync_time_controls_from_active_panel()
        self.active_panel.set_zoom_mode("auto")
        self.cursor_label.setText("当前图已切换，可在左侧为该图重新选择信号")

    def _on_subwindow_activated(self, subwindow: QMdiSubWindow | None) -> None:
        if subwindow is None:
            return
        panel = subwindow.widget()
        if isinstance(panel, BasePlotPanel) and panel in self.panels and panel is not self.active_panel:
            self.set_active_panel(panel)

    def _position_new_panel(self, subwindow: PlotSubWindow, mode: str) -> None:
        viewport_size = self.mdi_area.viewport().size()
        viewport_width = max(viewport_size.width(), 900)
        viewport_height = max(viewport_size.height(), 700)
        width = min(max(viewport_width - 80, 640), 1000)
        height = 460 if mode == "robot" else 420 if mode == "xyz" else 320
        height = min(height, viewport_height - 40)
        offset = 36 * (len(self.panels) - 1)
        x = min(offset, max(viewport_width - width - 20, 0))
        y = min(offset, max(viewport_height - height - 20, 0))
        subwindow.resize(width, height)
        subwindow.move(x, y)

    def _sync_tree_from_active_panel(self) -> None:
        if self.active_panel is None:
            return

        self._suppress_item_changed = True
        selected = set(self.active_panel.selected_signal_ids)
        for signal_id, item in self.signal_item_map.items():
            if item.isDisabled():
                continue
            item.setCheckState(0, Qt.CheckState.Checked if signal_id in selected else Qt.CheckState.Unchecked)
        self._suppress_item_changed = False

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._suppress_item_changed:
            return
        signal_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not signal_id or self.active_panel is None:
            return

        checked = item.checkState(0) == Qt.CheckState.Checked
        selection = list(self.active_panel.selected_signal_ids)
        limit = self.active_panel.selection_limit

        if checked:
            if signal_id not in selection:
                if limit is not None and len(selection) >= limit:
                    removed_signal_id = selection.pop(0)
                    removed_item = self.signal_item_map.get(removed_signal_id)
                    if removed_item is not None:
                        self._suppress_item_changed = True
                        removed_item.setCheckState(0, Qt.CheckState.Unchecked)
                        self._suppress_item_changed = False
                selection.append(signal_id)
        else:
            selection = [current for current in selection if current != signal_id]

        self.active_panel.selected_signal_ids = selection
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.active_panel.reset_view()

    def clear_selection(self) -> None:
        if self.active_panel is None:
            return
        self.active_panel.selected_signal_ids = []
        self._suppress_item_changed = True

        for item in self.signal_item_map.values():
            if not item.isDisabled():
                item.setCheckState(0, Qt.CheckState.Unchecked)

        self._suppress_item_changed = False
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.cursor_label.setText("未选择曲线")

    def reset_view(self) -> None:
        if self.parsed_log is None or self.active_panel is None:
            return
        self.active_panel.reset_view()

    def apply_time_range(self) -> None:
        if self.parsed_log is None or self.active_panel is None:
            return
        if not self.active_panel.supports_time_range:
            return

        start_seconds = float(self.time_start_input.value())
        end_seconds = float(self.time_end_input.value())
        if start_seconds >= end_seconds:
            QMessageBox.warning(self, "时间区间无效", "Start 必须小于 End。")
            return

        self.active_panel.set_time_range(start_seconds, end_seconds)
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.active_panel.reset_view()
        self.cursor_label.setText(f"已应用时间区间: {start_seconds:.3f}s - {end_seconds:.3f}s")

    def clear_time_range(self) -> None:
        if self.parsed_log is None or self.active_panel is None:
            return
        if not self.active_panel.supports_time_range:
            return

        self.active_panel.clear_time_range()
        self._sync_time_controls_from_active_panel()
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.active_panel.reset_view()
        self.cursor_label.setText("已恢复全部时间样本")

    def _sync_time_controls_from_active_panel(self) -> None:
        controls = (
            self.time_start_input,
            self.time_end_input,
            self.apply_time_range_button,
            self.clear_time_range_button,
        )
        if self.parsed_log is None or self.active_panel is None or not self.active_panel.supports_time_range:
            for control in controls:
                control.setEnabled(False)
            self.time_start_input.setRange(0.0, 0.0)
            self.time_end_input.setRange(0.0, 0.0)
            self.time_start_input.setValue(0.0)
            self.time_end_input.setValue(0.0)
            return

        max_seconds = float(self.parsed_log.time_seconds[-1])
        for spin_box in (self.time_start_input, self.time_end_input):
            spin_box.setEnabled(True)
            spin_box.setRange(0.0, max_seconds)
        self.apply_time_range_button.setEnabled(True)
        self.clear_time_range_button.setEnabled(True)

        if self.active_panel.time_range is None:
            start_seconds, end_seconds = 0.0, max_seconds
        else:
            start_seconds, end_seconds = self.active_panel.time_range

        self.time_start_input.setValue(start_seconds)
        self.time_end_input.setValue(end_seconds)

    def apply_filter(self, text: str) -> None:
        query = text.strip().lower()

        def filter_item(item: QTreeWidgetItem) -> bool:
            own_text = f"{item.text(0)} {item.toolTip(0)}".lower()
            child_visible = False

            for child_index in range(item.childCount()):
                child_visible = filter_item(item.child(child_index)) or child_visible

            visible = not query or query in own_text or child_visible
            item.setHidden(not visible)

            if query and child_visible:
                item.setExpanded(True)

            return visible

        for index in range(self.tree.topLevelItemCount()):
            filter_item(self.tree.topLevelItem(index))

    def _update_panel_status(self, panel: BasePlotPanel, text: str) -> None:
        if panel is self.active_panel:
            self.cursor_label.setText(text)

    def _on_event_row_activated(self, row: int, column: int) -> None:
        del column
        item = self.event_table.item(row, 0)
        if item is None:
            return

        sample_index = item.data(Qt.ItemDataRole.UserRole)
        if sample_index is None:
            return
        self.current_sample_index = int(sample_index)

        event = self.detected_events[row] if row < len(self.detected_events) else None
        for panel in self.panels:
            panel.focus_sample_index(int(sample_index))

        if event is not None:
            self.cursor_label.setText(
                f"事件: {event.time_seconds:.3f}s | {event.signal_name} "
                f"{event.previous_value:g} -> {event.current_value:g}"
            )

    def _sync_plot_cursors(self, source_panel: Plot2DPanel, index: int | None) -> None:
        if index is not None:
            self.current_sample_index = int(index)
        for panel in self.panels:
            if panel is source_panel:
                continue
            panel.sync_sample_index(index)


def launch_app(initial_path: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_path=initial_path)
    window.show()
    return app.exec()
