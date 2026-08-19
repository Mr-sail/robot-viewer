# 机器人状态曲线查看器

一个用于查看机器人运行日志的桌面工具，支持在 Windows 和 Ubuntu 上使用。它面向机器人调试、现场排障、工艺分析和研发验证人员，用来快速查看日志中的关节、TCP、状态码、报警码等时序数据。

本工具适合读取“XML 信号定义 + ID 表头 + 采样数据”的 `txt` 日志文件，并把日志字段组织成可搜索、可勾选的信号树。

## 主要功能

- 打开单个机器人日志 `txt` 文件
- 自动解析 XML 信号定义和 `ID` 数据列映射
- 左侧按目录展示信号字段，支持搜索和勾选
- 右侧支持多个独立图面板
- 支持 `X-T` 曲线、`X-Y` 轨迹图、`XYZ` 三维轨迹图
- 支持鼠标缩放、平移、重置视图
- 支持输入相对时间区间，只查看指定时间段内的数据
- 支持多图之间同步鼠标光标位置
- 支持键盘左右方向键逐采样点查看数据
- 支持在 2D 图中点击两个点，显示两点时间差和 4ms 周期数
- 加载文件时显示解析进度
- 自动识别 `ErrCode` 变化事件，并支持双击事件跳转到对应采样点
- 对异常数据行进行跳过统计，不影响其余数据查看

## 典型使用流程

1. 启动程序。
2. 点击 `打开文件`，选择机器人日志 `txt`。
3. 等待加载进度完成。
4. 在左侧字段树中搜索并勾选信号。
5. 在右侧图面板中查看曲线。
6. 如需局部分析，在 `时间区间` 中输入 Start / End 秒数并应用。
7. 如需逐周期查看数据，先点击当前 2D 图，再使用键盘 `←` / `→` 移动光标。
8. 如需测量时间差，在 2D 图中依次点击两个采样点，底部状态栏会显示时间差和周期数。
9. 如需查看报警变化，点击底部 `显示事件`，展开 `ErrCode` 事件列表。
10. 双击事件行，图中的光标会跳转到对应采样点附近。

## 图面板说明

### X-T 图

用于查看一个或多个信号随时间变化的曲线。横轴为相对日志起始时间，单位为秒。

适合查看：

- 关节位置、速度、加速度
- TCP 位姿
- 状态字段变化
- 报警码变化

### X-Y 图

用于查看两个信号之间的关系。勾选两个信号后，第一个作为 X，第二个作为 Y。

适合查看：

- TCP X/Y 平面轨迹
- 两个变量之间的关联关系

### XYZ 图

用于查看三个信号组成的三维轨迹。勾选三个信号后，按选择顺序作为 X、Y、Z。

使用 XYZ 图需要安装 `PyOpenGL`，完整依赖安装方式见下方安装说明。

## 逐周期查看与时间差测量

在任意 2D 图中，可以使用键盘和鼠标进行细节分析：

- `←`：光标向前移动 1 个采样点。
- `→`：光标向后移动 1 个采样点。
- 鼠标左键点击第一个点，再点击第二个点：底部状态栏显示两点时间差。

时间差显示为整数毫秒，并按 `4ms/周期` 计算周期数。

示例：

```text
两点时间差: 128 ms | 32 个周期(4ms/周期)
```

如果当前日志采样周期不是 4ms，周期数仅作为参考。

## 时间区间

文件加载后，顶部会显示 Start / End 输入框，单位为秒。

- `应用时间区间`：当前图只显示该时间段内的数据。
- `全部时间`：恢复当前图的完整时间范围。
- 每个图面板可以保留自己的时间区间设置。

时间区间用于快速聚焦某段动作、报警前后、暂停恢复等局部过程。

## ErrCode 事件

底部事件区域默认隐藏，只保留 `显示事件` 按钮和事件数量。

当前版本只识别字段名或路径节点为 `ErrCode` 的字段，避免把状态字、端口位、跟踪误差等高频变化字段误当成事件。

事件规则：

- `ErrCode` 从 0 变为非 0
- `ErrCode` 在非 0 值之间变化
- `ErrCode` 从非 0 恢复为 0

双击事件行后，所有 2D 图会同步显示该采样点的光标；X-T 图会自动缩放到事件附近。

## 安装

先进入项目目录：

```bash
cd Escope-viewer
```

完整桌面版依赖：

```bash
python3 -m pip install -r requirements.txt
```

Windows 下：

```powershell
py -m pip install -r requirements.txt
```

如果只需要运行解析层测试，可以安装最小依赖：

```bash
python3 -m pip install -r requirements-parser.txt
python3 -m unittest tests.test_parser tests.test_events
```

## 运行

不带文件启动：

```bash
python3 -m app
```

Windows 下：

```powershell
py -m app
```

启动时直接打开文件：

```bash
python3 -m app path/to/log.txt
```

Windows 下：

```powershell
py -m app .\path\to\log.txt
```

## 支持的日志格式

当前版本支持如下结构：

1. 文件前半段为 XML 信号定义。
2. 中间使用一行纯 `*` 作为分隔。
3. 分隔线下一行是 `ID` 表头。
4. 后续每行是一个采样点。
5. 第一列为时间戳，格式为 `YYYYMMDDHHMMSSmmm`。
6. 后续列为各个信号值，列顺序与 `ID` 表头对应。

示例结构：

```text
<?xml version="1.0"?>
<module>
  ...
</module>
********************
ID    001001001001    001008001001
20260301131750100    1.0    0
20260301131750104    1.1    1001
```

## 注意事项

- 大日志加载需要一定时间，加载过程中会显示进度。
- 少量坏行会被跳过，并在状态栏中显示跳过数量。
- 当前一次只打开一个日志文件。
- 当前事件识别只针对 `ErrCode`，其他状态字段暂不自动生成事件。
- 如果不需要三维图，也可以不使用 `XYZ` 面板。

## 生成 Ubuntu AppImage

在项目根目录执行以下命令。构建机需要 Python；目标 Ubuntu 不需要安装 Python。

```bash
cd /home/zhuyufan/Desktop/Public/Escope
python3 -m pip install -r requirements-build.txt
curl -L -o appimagetool \
  https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
echo "a6d71e2b6cd66f8e8d16c37ad164658985e0cf5fcaa950c90a482890cb9d13e0  appimagetool" | sha256sum --check
chmod +x appimagetool
./packaging/build_appimage.sh
```

成功后只需交付下面这个文件：

```text
release/Escope.AppImage
```

不要单独交付 `dist/Escope/Escope`，它依赖同目录下的 `_internal` 文件。目标机器如果没有 FUSE，可以使用：

```bash
APPIMAGE_EXTRACT_AND_RUN=1 ./Escope.AppImage
```

如果 xacro 文件使用了 ROS 的 `$(find package_name)`，还需要把 ROS 包资源准备到 AppImage 中。当前脚本默认打包 `estun_description` 和 `estun_hardware`：

```bash
ROS_INSTALL_PREFIX=/home/zhuyufan/estun_ws/install \
  ./packaging/prepare_ros_bundle.sh
```

如果模型还依赖其他 ROS 包，可以指定包名：

```bash
ROS_INSTALL_PREFIX=/path/to/ros_ws/install \
ROS_PACKAGES="estun_description estun_hardware another_description" \
  ./packaging/prepare_ros_bundle.sh
```

该步骤只复制 `share/<package>` 资源，不会把本机 ROS 工作区或构建产物提交到 Git。准备完成后，再执行下面的 AppImage 构建命令；AppRun 会自动设置内置的 ROS 包搜索路径。

如果需要兼容 Ubuntu 20.04，可以在 Ubuntu 22.04 主机上使用 Docker 构建，不需要另一台电脑：

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker

./packaging/build_ubuntu20_appimage.sh
```

构建结果为 `release/ubuntu20/Escope-Ubuntu20-x86_64.AppImage`。该版本应以 Ubuntu 20.04 的 glibc 作为兼容基线，建议在干净的 Ubuntu 20.04 虚拟机中最终验证。

如果 Docker Hub 无法访问，可指定一个可访问的 Python 3.10 基础镜像和 Debian 镜像源：

```bash
UBUNTU20_IMAGE=<可访问的镜像仓库>/library/python:3.10-slim-bullseye \
DEBIAN_MIRROR=<可访问的 Debian 镜像>/debian \
DEBIAN_SECURITY_MIRROR=<可访问的 Debian 安全镜像>/debian-security \
PIP_INDEX_URL=<可访问的 PyPI 镜像>/simple \
APPIMAGETOOL_SHA256=<已验证的 appimagetool SHA-256> \
  ./packaging/build_ubuntu20_appimage.sh
```

## 测试

安装开发依赖并运行 lint 与测试：

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m flake8 app tests launcher.py ament_index_python
python3 -m unittest discover -s tests
bash -n packaging/*.sh
```

Windows 下：

```powershell
py -m unittest discover -s tests
```

测试样例位于 `tests/fixtures/sample_log.txt`。
