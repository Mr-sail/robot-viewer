from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.events import EventDetectionSummary  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.models import LogEvent  # noqa: E402


class MainWindowLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_removing_panel_unregisters_its_drop_targets(self) -> None:
        initial_target_count = len(self.window._drop_targets)
        self.window.add_panel("xy")
        panel = self.window.active_panel

        self.assertIsNotNone(panel)
        self.assertGreater(len(self.window._drop_targets), initial_target_count)

        self.window.remove_panel(panel)

        self.assertEqual(len(self.window._drop_targets), initial_target_count)

    def test_event_summary_reports_truncation(self) -> None:
        event = LogEvent(
            sample_index=1,
            time_seconds=0.004,
            time_raw="20260301131750104",
            signal_id="err",
            signal_name="ErrCode",
            signal_path="ER / Status / ErrCode",
            previous_value=0.0,
            current_value=1.0,
            event_type="报警/错误变化",
        )
        summary = EventDetectionSummary(
            total_candidates=5,
            per_signal_truncated_count=4,
            global_truncated_count=0,
            returned_count=1,
        )

        self.window._populate_event_table([event], summary)

        self.assertEqual(self.window.event_summary_label.text(), "事件: 1/5 (已省略 4)")
        self.assertIn("已省略 4", self.window.event_table.toolTip())


if __name__ == "__main__":
    unittest.main()
