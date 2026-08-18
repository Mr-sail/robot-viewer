from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.robot_model import RobotModelError, load_robot_model


SIMPLE_URDF = """\
<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link" />
  <link name="link1" />
  <link name="tool0" />
  <joint name="joint1" type="revolute">
    <parent link="base_link" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
  <joint name="joint2" type="fixed">
    <parent link="link1" />
    <child link="tool0" />
    <origin xyz="1 0 0" rpy="0 0 0" />
  </joint>
</robot>
"""

MESH_URDF = """\
<?xml version="1.0"?>
<robot name="mesh_arm">
  <link name="base_link">
    <visual name="base_visual">
      <origin xyz="0 0 1" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/base.stl" scale="0.001 0.002 0.003" />
      </geometry>
    </visual>
  </link>
  <link name="tool0" />
  <joint name="joint1" type="fixed">
    <parent link="base_link" />
    <child link="tool0" />
    <origin xyz="1 0 0" rpy="0 0 0" />
  </joint>
</robot>
"""

PACKAGE_MESH_URDF = """\
<?xml version="1.0"?>
<robot name="package_mesh_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://my_robot/meshes/base.stl" />
      </geometry>
    </visual>
  </link>
  <link name="tool0" />
  <joint name="joint1" type="fixed">
    <parent link="base_link" />
    <child link="tool0" />
  </joint>
</robot>
"""

SIMPLE_XACRO = """\
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_arm">
  <xacro:property name="joint_offset" value="1.25" />
  <link name="base_link" />
  <link name="tool0" />
  <joint name="joint1" type="fixed">
    <parent link="base_link" />
    <child link="tool0" />
    <origin xyz="${joint_offset} 0 0" rpy="0 0 0" />
  </joint>
</robot>
"""


class RobotModelTests(unittest.TestCase):
    def test_load_robot_model_expands_xacro(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.xacro"
            path.write_text(SIMPLE_XACRO, encoding="utf-8")

            model = load_robot_model(path)

        self.assertEqual(model.name, "xacro_arm")
        np.testing.assert_allclose(model.joints[0].origin_xyz, np.array([1.25, 0.0, 0.0]))

    def test_load_robot_model_from_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.urdf"
            path.write_text(SIMPLE_URDF, encoding="utf-8")

            model = load_robot_model(path)

        self.assertEqual(model.name, "simple_arm")
        self.assertEqual(model.root_link, "base_link")
        self.assertEqual([joint.name for joint in model.movable_joints], ["joint1"])

    def test_compute_segments_applies_joint_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simple.urdf"
            path.write_text(SIMPLE_URDF, encoding="utf-8")
            model = load_robot_model(path)

        segments = model.compute_segments({"joint1": np.pi / 2.0})

        self.assertEqual(len(segments), 2)
        first_segment_end = segments[0][1]
        second_segment_end = segments[1][1]
        np.testing.assert_allclose(first_segment_end, np.array([1.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(second_segment_end, np.array([1.0, 1.0, 0.0]), atol=1e-6)

    def test_invalid_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "broken.urdf"
            path.write_text("<robot>", encoding="utf-8")

            with self.assertRaises(RobotModelError):
                load_robot_model(path)

    def test_load_robot_model_parses_visual_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "robot.urdf"
            path.write_text(MESH_URDF, encoding="utf-8")

            model = load_robot_model(path)

        self.assertEqual(len(model.visual_meshes), 1)
        mesh = model.visual_meshes[0]
        self.assertEqual(mesh.name, "base_visual")
        self.assertEqual(mesh.link_name, "base_link")
        self.assertEqual(mesh.path, (Path(tmpdir) / "meshes" / "base.stl").resolve())
        np.testing.assert_allclose(mesh.origin_xyz, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(mesh.scale, np.array([0.001, 0.002, 0.003]))

    def test_package_mesh_path_resolves_from_urdf_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = Path(tmpdir) / "ws" / "src" / "my_robot"
            urdf_dir = package_dir / "urdf"
            mesh_dir = package_dir / "meshes"
            urdf_dir.mkdir(parents=True)
            mesh_dir.mkdir(parents=True)
            mesh_path = mesh_dir / "base.stl"
            mesh_path.write_text("", encoding="utf-8")
            path = urdf_dir / "robot.urdf"
            path.write_text(PACKAGE_MESH_URDF, encoding="utf-8")

            model = load_robot_model(path)

        self.assertEqual(model.visual_meshes[0].path, mesh_path.resolve())


if __name__ == "__main__":
    unittest.main()
