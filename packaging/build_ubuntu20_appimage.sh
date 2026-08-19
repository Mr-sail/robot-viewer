#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$project_root/release/ubuntu20"
output_file="$output_dir/Escope-Ubuntu20-x86_64.AppImage"
image_tag="escope-appimage-ubuntu20:local"
base_image="${UBUNTU20_IMAGE:-python:3.10-slim-bullseye}"
debian_mirror="${DEBIAN_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian}"
debian_security_mirror="${DEBIAN_SECURITY_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian-security}"
pip_index_url="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
appimagetool_sha256="${APPIMAGETOOL_SHA256:-a6d71e2b6cd66f8e8d16c37ad164658985e0cf5fcaa950c90a482890cb9d13e0}"
container_id=""

cleanup() {
    if [[ -n "$container_id" ]]; then
        docker rm -f "$container_id" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
    echo "找不到 docker。请先在 Ubuntu 22.04 安装 Docker。" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "当前用户无法连接 Docker。请执行 sudo usermod -aG docker \"\$USER\" 后重新登录。" >&2
    exit 1
fi

mkdir -p "$output_dir"
DOCKER_BUILDKIT=0 docker build \
    --file "$project_root/packaging/Dockerfile.ubuntu20" \
    --build-arg "UBUNTU20_IMAGE=$base_image" \
    --build-arg "DEBIAN_MIRROR=$debian_mirror" \
    --build-arg "DEBIAN_SECURITY_MIRROR=$debian_security_mirror" \
    --build-arg "PIP_INDEX_URL=$pip_index_url" \
    --build-arg "APPIMAGETOOL_SHA256=$appimagetool_sha256" \
    --target builder \
    --tag "$image_tag" \
    "$project_root"

container_id="$(docker create "$image_tag")"
docker cp "$container_id:/src/release/Escope.AppImage" "$output_file"
chmod +x "$output_file"

echo "已生成: $output_file"
