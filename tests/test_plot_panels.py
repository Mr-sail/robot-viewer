import unittest
from pathlib import Path

import numpy as np

from app.models import LogFileMeta, ParsedLog, SignalNode
from app.plot_panels import Plot2DPanel, RobotPosePanel
from app.robot_model import RobotJoint, RobotModel


class XtDisplaySeriesTests(unittest.TestCase):
    def test_raw_series_is_unchanged(self) -> None:
        series = np.array([1.0, 3.5, 7.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "raw")

        np.testing.assert_allclose(derived, series)

    def test_first_difference_keeps_alignment_with_leading_nan(self) -> None:
        series = np.array([2.0, 5.0, 9.5, 10.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "diff1")

        self.assertTrue(np.isnan(derived[0]))
        np.testing.assert_allclose(derived[1:], np.array([3.0, 4.5, 0.5], dtype=float))

    def test_second_difference_keeps_alignment_with_two_leading_nans(self) -> None:
        series = np.array([1.0, 4.0, 9.0, 16.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "diff2")

        self.assertTrue(np.isnan(derived[0]))
        self.assertTrue(np.isnan(derived[1]))
        np.testing.assert_allclose(derived[2:], np.array([2.0, 2.0], dtype=float))


class XyNearestSampleTests(unittest.TestCase):
    def test_ignores_non_finite_pairs(self) -> None:
        local_index = Plot2DPanel._nearest_xy_local_index(
            np.array([0.0, np.nan, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            2.0,
            2.0,
        )

        self.assertEqual(local_index, 2)

    def test_returns_none_without_finite_pairs(self) -> None:
        local_index = Plot2DPanel._nearest_xy_local_index(
            np.array([np.nan, np.inf]),
            np.array([0.0, 1.0]),
            0.0,
            0.0,
        )

        self.assertIsNone(local_index)


class RobotSignalMappingTests(unittest.TestCase):
    def test_does_not_assign_velocity_from_one_joint_to_another_joint(self) -> None:
        movable_joints = [
            RobotJoint(
                name=f"joint{index}",
                joint_type="revolute",
                parent="base" if index == 1 else "link1",
                child=f"link{index}",
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
                axis=np.array([0.0, 0.0, 1.0]),
            )
            for index in (1, 2)
        ]
        robot_model = RobotModel(
            name="two_joint_arm",
            root_link="base",
            tool_link="link2",
            joints=movable_joints,
            links=("base", "link1", "link2"),
            children_by_link={"base": [movable_joints[0]], "link1": [movable_joints[1]]},
        )
        signals = [
            SignalNode(
                signal_id="j1_pos",
                name="J1",
                path_parts=("ER", "Joint", "Pos", "J1"),
                full_path="ER / Joint / Pos / J1",
                column_index=0,
                available=True,
            ),
            SignalNode(
                signal_id="j1_vel",
                name="J1",
                path_parts=("ER", "Joint", "Vel", "J1"),
                full_path="ER / Joint / Vel / J1",
                column_index=1,
                available=True,
            ),
        ]
        parsed_log = ParsedLog(
            meta=LogFileMeta(
                path=Path("test.txt"),
                sample_count=1,
                field_count=2,
                start_time_raw="20260301131750100",
                end_time_raw="20260301131750100",
            ),
            time_raw=np.array(["20260301131750100"]),
            time_seconds=np.array([0.0]),
            signals_by_id={"j1_pos": np.array([10.0]), "j1_vel": np.array([90.0])},
            signals=signals,
        )

        mapping = RobotPosePanel._build_joint_signal_map(robot_model, parsed_log)

        self.assertEqual(mapping, {"joint1": "j1_pos"})


if __name__ == "__main__":
    unittest.main()
