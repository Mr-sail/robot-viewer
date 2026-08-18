#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_prefix="${ROS_INSTALL_PREFIX:-}"
bundle_dir="$project_root/packaging/ros_bundle"

if [[ -z "$source_prefix" ]]; then
    echo "请设置 ROS_INSTALL_PREFIX，例如: ROS_INSTALL_PREFIX=~/estun_ws/install" >&2
    exit 1
fi
source_prefix="$(cd "$source_prefix" && pwd)"

if [[ ! -d "$source_prefix" ]]; then
    echo "ROS 安装前缀不存在: $source_prefix" >&2
    exit 1
fi

read -r -a packages <<< "${ROS_PACKAGES:-estun_description estun_hardware}"
rm -rf "$bundle_dir"
mkdir -p "$bundle_dir/share"

for package_name in "${packages[@]}"; do
    package_share="$source_prefix/$package_name/share/$package_name"
    if [[ ! -d "$package_share" ]]; then
        echo "找不到 ROS 包: $package_name ($package_share)" >&2
        exit 1
    fi
    mkdir -p "$bundle_dir/share/$package_name"
    cp -a "$package_share/." "$bundle_dir/share/$package_name/"
done

echo "已准备 ROS 资源包: $bundle_dir"
du -sh "$bundle_dir"

