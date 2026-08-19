#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-python3}"
app_name="Escope"
app_dir="$project_root/build/AppDir"
release_dir="$project_root/release"
appimage_path="$release_dir/Escope.AppImage"
appimage_tool="${APPIMAGETOOL:-}"
runtime_file="${APPIMAGE_RUNTIME:-}"

if [[ -z "$appimage_tool" ]]; then
    appimage_tool="$(command -v appimagetool || true)"
fi
if [[ -z "$appimage_tool" && -x "$project_root/appimagetool" ]]; then
    appimage_tool="$project_root/appimagetool"
fi
if [[ -z "$runtime_file" && -f "$project_root/runtime-x86_64" ]]; then
    runtime_file="$project_root/runtime-x86_64"
fi

command -v "$python_bin" >/dev/null 2>&1 || {
    echo "找不到 Python: $python_bin" >&2
    exit 1
}

"$python_bin" -m PyInstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "$app_name" \
    --paths "$project_root" \
    --exclude-module matplotlib \
    --exclude-module scipy \
    --exclude-module jupyter \
    --exclude-module jupyter_rfb \
    --collect-submodules xacro \
    --collect-submodules ament_index_python \
    "$project_root/launcher.py"

rm -rf "$app_dir"
mkdir -p "$app_dir/usr/bin" "$app_dir/usr/share/applications"
cp -a "$project_root/dist/$app_name/." "$app_dir/usr/bin/"
if [[ -d "$project_root/packaging/ros_bundle" ]]; then
    mkdir -p "$app_dir/usr/share/ros"
    cp -a "$project_root/packaging/ros_bundle/." "$app_dir/usr/share/ros/"
fi
cp "$project_root/packaging/escope.desktop" "$app_dir/usr/share/applications/escope.desktop"
cp "$project_root/packaging/escope.desktop" "$app_dir/escope.desktop"
cp "$project_root/packaging/escope.svg" "$app_dir/escope.svg"

cat > "$app_dir/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
app_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -d "$app_dir/usr/share/ros" ]]; then
    export ESCOPE_ROS_PREFIX_PATH="$app_dir/usr/share/ros"
fi
exec "$app_dir/usr/bin/Escope" "$@"
EOF
chmod +x "$app_dir/AppRun"

if [[ -n "$appimage_tool" ]]; then
    mkdir -p "$release_dir"
    runtime_args=()
    if [[ -n "$runtime_file" ]]; then
        runtime_args=(--runtime-file "$runtime_file")
    elif runtime_offset="$("$appimage_tool" --appimage-offset 2>/dev/null)" \
        && [[ "$runtime_offset" =~ ^[0-9]+$ ]]; then
        runtime_file="$project_root/build/runtime-x86_64"
        dd if="$appimage_tool" of="$runtime_file" bs=1 count="$runtime_offset" status=none
        runtime_args=(--runtime-file "$runtime_file")
    fi
    APPIMAGE_EXTRACT_AND_RUN=1 "$appimage_tool" "${runtime_args[@]}" "$app_dir" "$appimage_path"
    rm -rf "$app_dir" "$project_root/dist/$app_name"
    echo "已生成唯一交付文件: $appimage_path"
else
    echo "PyInstaller 产物已生成: $project_root/dist/$app_name"
    echo "AppDir 已生成: $app_dir"
    echo "未找到 appimagetool，无法自动生成 AppImage。"
    exit 2
fi
