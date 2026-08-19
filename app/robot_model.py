from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

import numpy as np


class RobotModelError(ValueError):
    """Raised when the robot model cannot be parsed or evaluated."""


@dataclass(frozen=True)
class RobotJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray

    @property
    def is_movable(self) -> bool:
        return self.joint_type in {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class RobotVisualMesh:
    name: str
    link_name: str
    filename: str
    path: Path | None
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    scale: np.ndarray

    @property
    def origin_transform(self) -> np.ndarray:
        return _compose_transform(self.origin_xyz, self.origin_rpy)


@dataclass
class RobotModel:
    name: str
    root_link: str
    tool_link: str
    joints: list[RobotJoint]
    links: tuple[str, ...]
    children_by_link: dict[str, list[RobotJoint]]
    visual_meshes: tuple[RobotVisualMesh, ...] = ()

    @property
    def movable_joints(self) -> list[RobotJoint]:
        return [joint for joint in self.joints if joint.is_movable]

    def compute_state(self, joint_values: dict[str, float]) -> "RobotPoseState":
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        joint_positions: list[np.ndarray] = []
        joint_transforms: dict[str, np.ndarray] = {}
        link_transforms: dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=float)}
        visited_links: set[str] = set()

        def walk(link_name: str, parent_transform: np.ndarray) -> None:
            if link_name in visited_links:
                raise RobotModelError(f"模型拓扑包含重复访问的 link: {link_name}")
            visited_links.add(link_name)
            parent_position = parent_transform[:3, 3].copy()
            for joint in self.children_by_link.get(link_name, []):
                origin_transform = _compose_transform(joint.origin_xyz, joint.origin_rpy)
                joint_frame = parent_transform @ origin_transform
                joint_transforms[joint.name] = joint_frame
                joint_position = joint_frame[:3, 3].copy()
                joint_positions.append(joint_position)
                segments.append((parent_position, joint_position))

                motion = _joint_motion_transform(joint, float(joint_values.get(joint.name, 0.0)))
                child_transform = joint_frame @ motion
                link_transforms[joint.child] = child_transform
                child_position = child_transform[:3, 3].copy()
                if not np.allclose(child_position, joint_position):
                    segments.append((joint_position, child_position))

                walk(joint.child, child_transform)

        walk(self.root_link, np.eye(4, dtype=float))
        tool_transform = link_transforms.get(self.tool_link, np.eye(4, dtype=float))
        return RobotPoseState(
            segments=segments,
            joint_positions=joint_positions,
            joint_transforms=joint_transforms,
            link_transforms=link_transforms,
            tool_transform=tool_transform,
        )

    def compute_segments(self, joint_values: dict[str, float]) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.compute_state(joint_values).segments


@dataclass
class RobotPoseState:
    segments: list[tuple[np.ndarray, np.ndarray]]
    joint_positions: list[np.ndarray]
    joint_transforms: dict[str, np.ndarray]
    link_transforms: dict[str, np.ndarray]
    tool_transform: np.ndarray


def _parse_vector(text: str | None, *, default: tuple[float, float, float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.array(default, dtype=float)

    parts = text.split()
    if len(parts) != 3:
        raise RobotModelError(f"expected 3 numbers, got: {text!r}")
    try:
        vector = np.array([float(part) for part in parts], dtype=float)
    except ValueError as exc:
        raise RobotModelError(f"invalid numeric vector: {text!r}") from exc
    if not np.isfinite(vector).all():
        raise RobotModelError(f"numeric vector must contain finite values: {text!r}")
    return vector


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / norm
    x_axis, y_axis, z_axis = axis
    sine = float(np.sin(angle))
    cosine = float(np.cos(angle))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x_axis * x_axis * one_minus_cosine,
                x_axis * y_axis * one_minus_cosine - z_axis * sine,
                x_axis * z_axis * one_minus_cosine + y_axis * sine,
            ],
            [
                y_axis * x_axis * one_minus_cosine + z_axis * sine,
                cosine + y_axis * y_axis * one_minus_cosine,
                y_axis * z_axis * one_minus_cosine - x_axis * sine,
            ],
            [
                z_axis * x_axis * one_minus_cosine - y_axis * sine,
                z_axis * y_axis * one_minus_cosine + x_axis * sine,
                cosine + z_axis * z_axis * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _compose_transform(translation: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_matrix_from_rpy(rpy)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _joint_motion_transform(joint: RobotJoint, value: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if joint.joint_type in {"revolute", "continuous"}:
        transform[:3, :3] = _axis_angle_rotation(joint.axis, value)
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = np.asarray(joint.axis, dtype=float) * value
    return transform


def _load_model_xml(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix != ".xacro":
        return path.read_text(encoding="utf-8", errors="replace")

    try:
        import xacro  # type: ignore
    except Exception:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        if "xacro:" in raw_text:
            raise RobotModelError(
                "当前安装包未包含 xacro 支持；请使用完整安装包、直接加载 URDF，"
                "或加载已经展开成普通 XML 的 xacro 文件。"
            )
        return raw_text

    try:
        document = xacro.process_file(str(path))
        return document.toxml()
    except Exception as exc:
        raise RobotModelError(f"xacro 展开失败: {exc}") from exc


def _candidate_package_roots(model_path: Path) -> list[Path]:
    roots = [model_path.parent, *model_path.parent.parents]
    for entry in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry).expanduser())
    return roots


def _candidate_package_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    for variable_name in ("ESCOPE_ROS_PREFIX_PATH", "AMENT_PREFIX_PATH"):
        for entry in os.environ.get(variable_name, "").split(os.pathsep):
            if entry.strip():
                prefixes.append(Path(entry).expanduser())
    return prefixes


def _resolve_package_mesh_path(package_name: str, relative_path: str, model_path: Path) -> Path | None:
    relative = Path(relative_path.lstrip("/"))
    for prefix in _candidate_package_prefixes():
        candidate = (prefix / "share" / package_name / relative).resolve()
        if candidate.exists():
            return candidate
    for root in _candidate_package_roots(model_path):
        if root.name == package_name:
            candidate = (root / relative).resolve()
            if candidate.exists():
                return candidate
        candidate = (root / package_name / relative).resolve()
        if candidate.exists():
            return candidate
    return None


def _resolve_mesh_path(filename: str, model_path: Path) -> Path | None:
    raw_filename = filename.strip()
    if not raw_filename:
        return None

    parsed = urlparse(raw_filename)
    if parsed.scheme in {"package", "model"}:
        package_name = unquote(parsed.netloc)
        relative_path = unquote(parsed.path.lstrip("/"))
        if package_name and relative_path:
            return _resolve_package_mesh_path(package_name, relative_path, model_path)
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme and parsed.scheme not in {"package", "model", "file"}:
        return None

    mesh_path = Path(unquote(raw_filename)).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = model_path.parent / mesh_path
    return mesh_path.resolve()


def _parse_visual_meshes(root: ET.Element, model_path: Path) -> tuple[RobotVisualMesh, ...]:
    visual_meshes: list[RobotVisualMesh] = []
    for link_element in root.findall("link"):
        link_name = (link_element.attrib.get("name") or "").strip()
        if not link_name:
            continue

        for index, visual_element in enumerate(link_element.findall("visual"), start=1):
            geometry_element = visual_element.find("geometry")
            if geometry_element is None:
                continue
            mesh_element = geometry_element.find("mesh")
            if mesh_element is None:
                continue
            filename = (mesh_element.attrib.get("filename") or mesh_element.attrib.get("url") or "").strip()
            if not filename:
                continue

            origin_element = visual_element.find("origin")
            origin_xyz = _parse_vector(
                origin_element.attrib.get("xyz") if origin_element is not None else None,
                default=(0.0, 0.0, 0.0),
            )
            origin_rpy = _parse_vector(
                origin_element.attrib.get("rpy") if origin_element is not None else None,
                default=(0.0, 0.0, 0.0),
            )
            scale = _parse_vector(mesh_element.attrib.get("scale"), default=(1.0, 1.0, 1.0))
            visual_name = (visual_element.attrib.get("name") or f"{link_name}_visual_{index}").strip()
            visual_meshes.append(
                RobotVisualMesh(
                    name=visual_name,
                    link_name=link_name,
                    filename=filename,
                    path=_resolve_mesh_path(filename, model_path),
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    scale=scale,
                )
            )
    return tuple(visual_meshes)


def _build_robot_graph(
    links: tuple[str, ...], joints: list[RobotJoint]
) -> tuple[str, str, list[RobotJoint], dict[str, list[RobotJoint]]]:
    link_names = set(links)
    if len(link_names) != len(links):
        raise RobotModelError("模型中的 link 名称必须唯一。")

    joint_names: set[str] = set()
    parent_by_child: dict[str, str] = {}
    children_by_link: dict[str, list[RobotJoint]] = {}
    for joint in joints:
        if joint.name in joint_names:
            raise RobotModelError(f"模型中的 joint 名称重复: {joint.name}")
        joint_names.add(joint.name)

        if joint.parent not in link_names:
            raise RobotModelError(f"joint {joint.name} 引用了不存在的 parent link: {joint.parent}")
        if joint.child not in link_names:
            raise RobotModelError(f"joint {joint.name} 引用了不存在的 child link: {joint.child}")
        if joint.child in parent_by_child:
            raise RobotModelError(f"link {joint.child} 由多个 joint 连接到父节点。")

        parent_by_child[joint.child] = joint.parent
        children_by_link.setdefault(joint.parent, []).append(joint)

    root_candidates = [link_name for link_name in links if link_name not in parent_by_child]
    if len(root_candidates) != 1:
        raise RobotModelError(
            f"模型必须恰好有一个根 link，当前找到 {len(root_candidates)} 个。"
        )
    root_link = root_candidates[0]

    ordered_joints: list[RobotJoint] = []
    visited_links: set[str] = set()
    link_depths: dict[str, int] = {root_link: 0}

    def visit(link_name: str) -> None:
        if link_name in visited_links:
            raise RobotModelError(f"模型拓扑包含环: {link_name}")
        visited_links.add(link_name)
        for joint in children_by_link.get(link_name, []):
            ordered_joints.append(joint)
            link_depths[joint.child] = link_depths[link_name] + 1
            visit(joint.child)

    visit(root_link)
    if visited_links != link_names:
        unreachable_links = ", ".join(sorted(link_names - visited_links))
        raise RobotModelError(f"模型拓扑不连通或包含环，无法从根节点访问: {unreachable_links}")

    leaf_links = {link_name for link_name in links if link_name not in children_by_link}
    preferred_tool_links = ("tool0", "tool", "flange", "ee_link", "end_effector")
    # Conventional end-effector names are more reliable than joint order in branched models.
    tool_link = next(
        (link_name for link_name in preferred_tool_links if link_name in leaf_links), None
    )
    if tool_link is None:
        tool_link = min(leaf_links, key=lambda link_name: (-link_depths[link_name], link_name))
    return root_link, tool_link, ordered_joints, children_by_link


def load_robot_model(path: str | Path) -> RobotModel:
    model_path = Path(path).expanduser().resolve()
    if not model_path.exists():
        raise RobotModelError(f"机器人模型不存在: {model_path}")
    if not model_path.is_file():
        raise RobotModelError(f"机器人模型不是有效文件: {model_path}")

    try:
        root = ET.fromstring(_load_model_xml(model_path))
    except OSError as exc:
        raise RobotModelError(f"机器人模型读取失败: {exc}") from exc
    except ET.ParseError as exc:
        raise RobotModelError(f"机器人模型 XML 解析失败: {exc}") from exc

    if root.tag != "robot":
        raise RobotModelError("模型根节点不是 <robot>，请提供 URDF 或可展开的 xacro 文件。")

    links = tuple(
        link.attrib.get("name", "").strip()
        for link in root.findall("link")
        if link.attrib.get("name", "").strip()
    )
    visual_meshes = _parse_visual_meshes(root, model_path)
    joints: list[RobotJoint] = []

    for joint_element in root.findall("joint"):
        name = (joint_element.attrib.get("name") or "").strip()
        joint_type = (joint_element.attrib.get("type") or "").strip()
        parent_link = (joint_element.findtext("parent", default="", namespaces={}) or "").strip()
        child_link = (joint_element.findtext("child", default="", namespaces={}) or "").strip()
        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        if parent_element is not None:
            parent_link = (parent_element.attrib.get("link") or parent_link).strip()
        if child_element is not None:
            child_link = (child_element.attrib.get("link") or child_link).strip()

        if not name or not joint_type or not parent_link or not child_link:
            raise RobotModelError("joint 定义缺少 name、type、parent 或 child。")

        origin_element = joint_element.find("origin")
        axis_element = joint_element.find("axis")
        origin_xyz = _parse_vector(
            origin_element.attrib.get("xyz") if origin_element is not None else None,
            default=(0.0, 0.0, 0.0),
        )
        origin_rpy = _parse_vector(
            origin_element.attrib.get("rpy") if origin_element is not None else None,
            default=(0.0, 0.0, 0.0),
        )
        axis = _parse_vector(
            axis_element.attrib.get("xyz") if axis_element is not None else None,
            default=(1.0, 0.0, 0.0),
        )
        if joint_type in {"revolute", "continuous", "prismatic"}:
            axis_norm = float(np.linalg.norm(axis))
            if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
                raise RobotModelError(f"可运动 joint {name} 的 axis 必须是非零有限向量。")
            axis = axis / axis_norm
        joints.append(
            RobotJoint(
                name=name,
                joint_type=joint_type,
                parent=parent_link,
                child=child_link,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )
        )

    if not joints:
        raise RobotModelError("模型中没有可用 joint 定义。")

    root_link, tool_link, ordered_joints, children_by_link = _build_robot_graph(links, joints)

    return RobotModel(
        name=(root.attrib.get("name") or model_path.stem).strip() or model_path.stem,
        root_link=root_link,
        tool_link=tool_link,
        joints=ordered_joints,
        links=links,
        children_by_link=children_by_link,
        visual_meshes=visual_meshes,
    )
