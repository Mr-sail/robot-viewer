from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

BRANCHED_URDF = """\
<?xml version="1.0"?>
<robot name="branched_arm">
  <link name="base_link" />
  <link name="tool0" />
  <link name="branch" />
  <link name="camera" />
  <joint name="tool_joint" type="fixed">
    <parent link="base_link" />
    <child link="tool0" />
    <origin xyz="1 0 0" />
  </joint>
  <joint name="branch_joint" type="fixed">
    <parent link="base_link" />
    <child link="branch" />
  </joint>
  <joint name="camera_joint" type="fixed">
    <parent link="branch" />
    <child link="camera" />
    <origin xyz="0 2 0" />
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
        self.assertEqual(model.tool_link, "tool0")
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

    def test_package_and_model_mesh_paths_resolve_from_prefix(self) -> None:
        for environment_variable in ("ESCOPE_ROS_PREFIX_PATH", "AMENT_PREFIX_PATH"):
            for scheme in ("package", "model"):
                with (
                    self.subTest(environment_variable=environment_variable, scheme=scheme),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root_dir = Path(tmpdir)
                    prefix = root_dir / "install"
                    mesh_path = prefix / "share" / "my_robot" / "meshes" / "base.stl"
                    mesh_path.parent.mkdir(parents=True)
                    mesh_path.write_text("", encoding="utf-8")
                    path = root_dir / "models" / "robot.urdf"
                    path.parent.mkdir()
                    path.write_text(
                        PACKAGE_MESH_URDF.replace("package://", f"{scheme}://"),
                        encoding="utf-8",
                    )

                    with mock.patch.dict(
                        "os.environ", {environment_variable: str(prefix)}, clear=True
                    ):
                        model = load_robot_model(path)

                    self.assertEqual(model.visual_meshes[0].path, mesh_path.resolve())

    def test_xacro_find_and_package_mesh_share_the_bundled_prefix(self) -> None:
        fragment = """\
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="bundled_robot">
    <link name="base_link">
      <visual><geometry><mesh filename="package://my_robot/meshes/base.stl" /></geometry></visual>
    </link>
    <link name="tool0" />
    <joint name="tool_joint" type="fixed">
      <parent link="base_link" /><child link="tool0" />
    </joint>
  </xacro:macro>
</robot>
"""
        model_xacro = """\
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="bundled">
  <xacro:include filename="$(find my_robot)/urdf/fragment.xacro" />
  <xacro:bundled_robot />
</robot>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            prefix = root_dir / "install"
            package_dir = prefix / "share" / "my_robot"
            mesh_path = package_dir / "meshes" / "base.stl"
            fragment_path = package_dir / "urdf" / "fragment.xacro"
            mesh_path.parent.mkdir(parents=True)
            fragment_path.parent.mkdir(parents=True)
            mesh_path.write_text("", encoding="utf-8")
            fragment_path.write_text(fragment, encoding="utf-8")
            path = root_dir / "robot.xacro"
            path.write_text(model_xacro, encoding="utf-8")

            with mock.patch.dict(
                "os.environ", {"ESCOPE_ROS_PREFIX_PATH": str(prefix)}, clear=True
            ):
                model = load_robot_model(path)

        self.assertEqual(model.name, "bundled")
        self.assertEqual(model.visual_meshes[0].path, mesh_path.resolve())

    def test_graph_validation_rejects_invalid_topologies(self) -> None:
        invalid_models = {
            "cycle": """
                <robot name="bad">
                  <link name="a"/><link name="b"/>
                  <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
                  <joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint>
                </robot>
            """,
            "disconnected": """
                <robot name="bad">
                  <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
                  <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
                  <joint name="cd" type="fixed"><parent link="c"/><child link="d"/></joint>
                </robot>
            """,
            "multiple parents": """
                <robot name="bad">
                  <link name="a"/><link name="b"/><link name="c"/>
                  <joint name="ac" type="fixed"><parent link="a"/><child link="c"/></joint>
                  <joint name="bc" type="fixed"><parent link="b"/><child link="c"/></joint>
                </robot>
            """,
            "missing link": """
                <robot name="bad">
                  <link name="a"/>
                  <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
                </robot>
            """,
            "duplicate joint name": """
                <robot name="bad">
                  <link name="a"/><link name="b"/><link name="c"/>
                  <joint name="joint" type="fixed"><parent link="a"/><child link="b"/></joint>
                  <joint name="joint" type="fixed"><parent link="b"/><child link="c"/></joint>
                </robot>
            """,
            "non-finite origin": """
                <robot name="bad">
                  <link name="a"/><link name="b"/>
                  <joint name="ab" type="fixed">
                    <parent link="a"/><child link="b"/><origin xyz="nan 0 0"/>
                  </joint>
                </robot>
            """,
        }

        for case_name, urdf in invalid_models.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "invalid.urdf"
                path.write_text(urdf, encoding="utf-8")

                with self.assertRaises(RobotModelError):
                    load_robot_model(path)

    def test_branched_model_prefers_named_tool_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "branched.urdf"
            path.write_text(BRANCHED_URDF, encoding="utf-8")
            model = load_robot_model(path)

        state = model.compute_state({})

        self.assertEqual(model.tool_link, "tool0")
        np.testing.assert_allclose(state.tool_transform[:3, 3], np.array([1.0, 0.0, 0.0]))

    def test_prismatic_axis_is_normalized(self) -> None:
        urdf = """
            <robot name="slider">
              <link name="base"/><link name="tool"/>
              <joint name="slide" type="prismatic">
                <parent link="base"/><child link="tool"/><axis xyz="2 0 0"/>
              </joint>
            </robot>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "slider.urdf"
            path.write_text(urdf, encoding="utf-8")
            model = load_robot_model(path)

        state = model.compute_state({"slide": 0.5})

        np.testing.assert_allclose(model.joints[0].axis, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(state.tool_transform[:3, 3], np.array([0.5, 0.0, 0.0]))

    def test_movable_joint_rejects_zero_axis(self) -> None:
        urdf = """
            <robot name="invalid_axis">
              <link name="base"/><link name="tool"/>
              <joint name="slide" type="prismatic">
                <parent link="base"/><child link="tool"/><axis xyz="0 0 0"/>
              </joint>
            </robot>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid_axis.urdf"
            path.write_text(urdf, encoding="utf-8")

            with self.assertRaises(RobotModelError):
                load_robot_model(path)


if __name__ == "__main__":
    unittest.main()
