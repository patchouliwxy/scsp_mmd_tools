# scsp_mmd_tools (v0.2.0 Preview)

**简体中文** | [English](README.en.md)

当前版本：**v0.2.0 预览版（Preview）** — 集成 **SCSP 角色建模（PMX）** 与 **歌曲动作/镜头导出（VMD）** 的桌面与命令行工具：
- **「角色 PMX」页签**：输入完整角色任务文件夹，在 Blender 后台自动生成干净无物理的 MMD 模型（`.pmx` + `textures`），**无需 Unity、PmxEditor 或第三方 Blender 插件**。
- **「动作 VMD」页签**：输入歌曲下载任务目录，直接转换为标准 30 FPS MMD 骨骼、表情、口型、视线与镜头 `.vmd`；内置纯 Python `BuiltinPoseBackend` Humanoid 解算器，**无需外置 Unity / PoseWorker 运行时**。

请保留 `scsp_mmd_tools.exe` 与 `_internal` 在同一文件夹，分发时复制整个发布目录。

## 运行环境

- **仅需使用「角色 PMX」页签 / CLI**：本机安装 **Blender >= 4.1**（已实测支持 **Blender 4.2 LTS** 与 **Blender 5.3.0**；启动时自动探测，也可通过 `SCSP_MMD_BLENDER` 指定）。源码版仅用 Python 标准库即可启动界面并运行角色转 PMX。
- **使用「动作 VMD」页签 / `motion_pipeline.py`**：可执行目录 `scsp_mmd_tools.exe` 已随包内置全部依赖；源码运行需额外安装 [requirements.txt](requirements.txt) 中的 **`numpy`**（BSD-3-Clause）与 **`UnityPy`**（MIT）。
- **零外部商业插件**：角色侧内置纯开源 **ufbx（MIT）** FBX 导入器（`Tools/ufbx_bridge.dll` + `Tools/ufbx_importer.py`）与纯 Python **PMX 2.0/2.1 读写器**（`Tools/pmx_io.py`）；动作侧内置基于 `Resources/humanoid_avatar.json`（纯数学骨骼层级与运动学参数表，不含网格/贴图）的纯 Python 解算器。

## 输入目录要求

不强制必须来自下载器（`job.json` 与 `manifest.json` 均为可选），手动整理的目录同样可用，但必须符合 SCSP FBX 命名、贴图与骨骼拓扑约定：

- 目录内需包含 `models/` 子目录（或通过 `job.json` 的 `model_files` 显式列出相对路径）及随附贴图；
- 仅识别 `M_Head_<数字ID>.fbx`（恰好 1 个，ID 需在 `Resources/character-master-20006200.json` 中）、`M_Dress_*.fbx`（恰好 1 个）、`M_Hair_*.fbx`（0 或 1 个）、`M_Acce_*.fbx`（0 或多个）。

## 界面使用

启动 `Releases/scsp_mmd_tools/scsp_mmd_tools.exe` 后可通过顶部切换栏在 **「角色 PMX」** 与 **「动作 VMD」** 两个独立页面之间切换（两页状态、队列与输出互不干扰，配置均自动持久化到 `studio_settings.json`）：

### 角色 PMX 页
1. 选择 Blender 可执行文件与输出目录（窗口顶部状态栏会实时显示探测到的 Blender 版本及内置 `ufbx FBX` / `PMX IO` 状态）。
2. 添加一个或多个下载任务目录：
   - **单项制作**：制作当前选中任务；**批量制作**：依次处理队列并在单项失败后继续下一项；**当前完成后停止**不会中断正在处理的模型。
   - 右侧详情支持设置可选**手工角色名**与多行 **PMX 说明**（默认留空，留空则不写入 PMX 说明；支持 `{name}`、`{romanized}`、`{id}`、`{character_id}` 占位符，可点“应用到选中 / 应用到全部 / 设为默认”保存到 `studio_settings.json`）。
3. 每个任务生成独立输出目录，不会覆盖已有成品；`_work` 保存中间 `.blend` 工程、配置、报告与日志，最终 PMX 请连同 `textures` 目录一起移动。

### 动作 VMD 页
1. 选择 **歌曲任务目录** 与 **输出目录**（**默认目标 PMX** 与 **自定义规则** 为可选项；默认目标 PMX 留空时按内置通用 45° A-pose MMD 骨架模板导出）。
2. 点击 **检查任务** 读取歌曲站位、时长与口型清单；可按选中站位单独指定目标 `.pmx`、清除回退通用模板或全选导出（单人曲自动兼容无 `_01` 编号的 `_exp.json` / `_root.unity3d`）。
3. 点击 **开始转换** 自动生成各站位 `.vmd`、公共原点相机 `*_camera.vmd`、`转换报告.json` 与 `使用说明.txt`。

## 处理流程与特性

处理模块：`ufbx` FBX 导入与材质/妆容整理 → 眼部与静态高光贴图合成 → 骨架、辅助权重合并与 45° A-pose → 角色比例、肤色烘焙与专属皮肤 Toon → 脸部圆柱自定义法向 → 按部件网格合并 → 62 项 MMD 形态键生成与映射 → 严格主体导出 → 纯 Python `pmx_io` 中心、足趾 IK、半标准控制骨与表情面板整理。

- **材质与肤色**：Blender 粗糙度为 `1`；PMX 默认扩散色 `(1,1,1)`、反射色 `(0,0,0)`、环境色 `(0.25,0.25,0.25)`、高光指数 `1`。依据角色主数据自动烘焙 `BodySkin` 与独立 `M_face` 皮肤色板并生成 `16×32` 皮肤专属 Toon 渐变；实际观感建议在目标渲染环境中手动检查。
- **配饰妆容与辅助层清理**：自动将配饰携带的 `M_face` 妆容贴图转移合并至脸部主材质，移除叠层脸膜、阴影代理 `*_shadow`、`closeface`、占位块、碰撞体及 `RoomShoes` 室内鞋（普通 `Shoes` 保留）。
- **眼睛与眼部特效**：高光静态合入主贴图并保留原始 Alpha（不恢复游戏动态 UV/高光位移）；按 Alpha 边界裁剪生成同心极坐标凹碗虹膜（`32` 径向分段 × `4` 环）并校准 1× 蛋状眼窝与球心眼骨支点（通过 `eye_hole_{L,R}` 边界映射生成，仅在遇到异常缺损眼眶时安全回退保留源几何并记录警告）。
- **表情形态键**：清理残缺/空源键（`口開け`、`左追加睫毛`、`右追加睫毛`、`tongue_up`），包含纯几何眼部扩展（`じと目`/`瞳小`/`瞳縦潰れ`/`びっくり`/`わぉ?!`/`なごみ`/`恐ろしい子！`）、脑内隐藏贴片特效（`星目`/`はぁと`/`はちゅ目`/`はぅ`）以及完整眉部（`個性`/`にこり`/`怒り`）与口型扩展（`ワ`/`にっこり`/`にやり`/`ω`/`ω□`/`▲`/`∧`/`□`/`ん`/`口角上げ`/`口角下げ`/`口横広げ`）。
- **体型配方与物理**：使用冻结角色主数据的身高、头型、胸围配合显式手臂 `0.95` / 手掌 `0.9` 配方（未恢复游戏原生公式）。主流程输出干净**无物理** PMX；如需原生刚体/关节物理，可对导出模型单独运行 `python Tools/pe_physics.py --input <角色.pmx> --output <物理版.pmx>`（支持 `--input-dir` / `--output-dir` 批量模式）。

## 源码运行、构建与测试

- 启动图形界面或仅使用 **「角色 PMX」** 页面 / `Tools/test_studio_pipeline.py` 时，仅依赖 Python 标准库（`>= 3.10`，界面需自带 Tkinter）与本机 Blender；
- 使用 **「动作 VMD」** 导出、`motion_pipeline.py`、运行 `tests/` 动作测试或打包完整发行版前，需先安装动作转换依赖（`pip install -r requirements.txt`，含 `numpy` 与 `UnityPy`）；Windows 下可直接双击或运行根目录的一键脚本 [Setup-Env.cmd](Setup-Env.cmd) 自动创建 `.venv` 并安装动作转换所需依赖（国内加速可加 `--mirror`，需打包 EXE 可加 `--build`）：

```powershell
# 一键安装动作转换环境（创建 .venv 并安装 numpy / UnityPy）
.\Setup-Env.cmd

# 启动图形界面（使用 Setup-Env.cmd 配置好的环境）
.\.venv\Scripts\pythonw.exe studio_gui.py

# 命令行：PMX 建模
python studio_pipeline.py inspect <任务文件夹>
python studio_pipeline.py build <任务文件夹> --output <输出根目录> --blender <Blender路径>
python studio_pipeline.py batch <任务1> <任务2> --output <输出根目录>

# 命令行：动作 / 相机转 VMD（默认使用内置纯 Python 解算器）
python motion_pipeline.py inspect <歌曲任务目录>
python motion_pipeline.py build <歌曲任务目录> --output <输出根目录>

# 运行单元测试
python -m unittest Tools/test_studio_pipeline.py -v
python -m unittest discover -s tests -v
```

### 从源码构建可执行包（Windows）

若从 GitHub 克隆源码（仓库默认不跟踪 `Releases/` 构建产物），请在已安装 `PyInstaller` 与 `requirements.txt`（`numpy`、`UnityPy`）的 Python 环境中打包生成独立目录：

```powershell
powershell -ExecutionPolicy Bypass -File build_studio.ps1 -Python <python路径> -CreateZip
```

产物将生成在 `Releases/scsp_mmd_tools/scsp_mmd_tools.exe`（同步保留既有 `Outputs`）。仓库已预编并提供 `Tools/ufbx_bridge.dll`，如需自行重编原生桥接库，可执行 `Tools/native/build_ufbx.ps1`（需 MinGW `gcc`）。

## 更新日志

### v0.2.0 Preview (2026-11-18)

- **集成动作 / 镜头转 VMD 功能与顶部双页签界面**：在 `studio_gui.py` 顶部新增 `角色 PMX` / `动作 VMD` 切换栏并保持与原有 PMX 处理流程完全隔离；`MotionTab` 独立复用 `studio_settings.json` 持久化工作路径，支持多站位独立指定目标 `.pmx` 或留空通用导出。
- **内置纯 Python Humanoid 动作解算器（移除外部 `PoseWorker` 依赖）**：新增 `BuiltinPoseBackend` 与纯数学运动学参数表 `Resources/humanoid_avatar.json`（仅 `46 KB`，不含网格与贴图）。
- **补全相机非线性变速、单人曲规则与延迟导入**：支持非 1.0 `moveSpeed` 幂次采样与提前结束 `easing` 曲线、兼容单人曲无 `_01` 编号的 `_exp.json`、补充 `requirements.txt` 并将动作模块改为延迟加载以保证纯标准库环境仍可启动 `角色 PMX` 页面。

### v0.1.4 Preview (2026-11-17)

- 修复牙饰与面饰兼容。

### v0.1.3 Preview (2026-11-17)

- 几何重叠自动消除。

### v0.1.2 Preview (2026-11-17)

- **修复部分电脑启动扫盘报错 `[WinError 87]` 的问题**：`studio_pipeline.py` 中的 `discover_default_blender()` 为 `is_dir()`、`glob()` 与 `is_file()` 增加 `OSError` 安全兜底，遇到未就绪虚拟盘、空光驱/读卡器或不可访问盘符时自动跳过而不崩溃；`studio_gui.py` 目录浏览逻辑同步补充异常捕获，`Tools/test_studio_pipeline.py` 新增对应回归测试。

### v0.1.1 Preview (2026-11-17)

- **修复 PMX 贴图路径写入绝对路径的问题**：`Tools/pe_core.py` 在收尾导出阶段不再将完整磁盘绝对路径写入 `.pmx`，统一改写为相对路径 `textures\<文件名>`，解决导入 MikuMikuDance（MMD）时因路径拼接失效导致的**模型全身纯白**问题，以及目录层级较深时超出 MMD/Windows 路径长度限制导致的**加载闪退**问题；同时确保移动模型文件夹或跨电脑分享后贴图依旧正常。
- **同步完善校验与测试**：`Tools/audit_pe_result.py` 改为相对于成品 `.pmx` 目录解析贴图并禁止残留绝对路径；`Tools/test_studio_pipeline.py` 新增相对贴图路径回归测试。

### v0.1 Preview (2026-11-16)

- 首个预览版发布，内置开源 `ufbx` FBX 导入器与纯 Python `pmx_io` 读写器，无外部插件依赖批量导出干净无物理 `.pmx`。

## 作者说明

本项目由 AI 辅助完成，部分处理为本人猜测进行的近似处理，不对导出产品有质量保证；有任何问题请提交 Issue 或在社交媒体私信我提交 Bug。

## 开源协议与免责声明

- 本项目代码基于 [GNU General Public License v3.0 (GPL-3.0)](LICENSE) 开源；内嵌 C FBX 解析库 [ufbx](ThirdParty/ufbx/LICENSE) 及运行时依赖 `UnityPy` 基于 MIT 协议提供，`numpy` 基于 BSD-3-Clause 协议提供，纯 Python PMX 读写、Humanoid 解算器与全部预处理工具均为独立开源实现。
- 本仓库不包含任何受版权保护的第三方游戏模型、贴图或音视频资产，仅用于个人非商业化格式研究与学习；涉及游戏角色元数据、骨骼及贴图请遵守原始版权方规定与使用条款。
