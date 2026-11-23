# scsp_mmd_tools (v0.2.2 Preview)

[简体中文](README.md) | **English**

Current release: **v0.2.2 Preview** — A unified desktop GUI and CLI pipeline combining **SCSP Character PMX Modeling** and **Song Motion / Camera VMD Conversion**:
- **Character PMX tab**: converts SCSP character assets into clean, physics-free MMD models (`.pmx` + `textures/`) in background Blender processes, **without requiring Unity, PmxEditor, or third-party Blender add-ons**.
- **Motion VMD tab**: converts song download folders into standard 30 FPS MMD body, expression, lip-sync, gaze, and camera `.vmd` files via a built-in pure-Python `BuiltinPoseBackend` Humanoid solver, **without requiring an external Unity / PoseWorker runtime**.

Keep `scsp_mmd_tools.exe` alongside `_internal/` when distributing or moving the packaged build.

## Requirements

- **Character PMX tab / CLI only**: **Blender >= 4.1** installed locally (tested on **Blender 4.2 LTS** and **Blender 5.3.0**; auto-detected at startup or specified via `SCSP_MMD_BLENDER`). Running the Character PMX workflow from source uses only the Python standard library.
- **Motion VMD tab / `motion_pipeline.py`**: the packaged `scsp_mmd_tools.exe` bundles all runtime dependencies. When running or packaging from source, install [requirements.txt](requirements.txt) (`numpy` under BSD-3-Clause and `UnityPy` under MIT).
- **Zero external commercial add-ons**: character import uses embedded open-source **ufbx (MIT)** (`Tools/ufbx_bridge.dll` + `Tools/ufbx_importer.py`) and pure-Python **PMX 2.0/2.1 I/O** (`Tools/pmx_io.py`); motion solving uses a pure-Python Humanoid solver backed by the mathematical kinematics table `Resources/humanoid_avatar.json` (no meshes or textures).

## Input Folder Structure

Folders do **not** strictly have to come from the downloader (`job.json` and `manifest.json` are both optional). Manually arranged folders also work as long as they follow SCSP FBX naming, texture, and rig/topology conventions:

- Include a `models/` subfolder containing the `.fbx` parts and accompanying textures (or declare relative paths via `model_files` inside an optional `job.json`).
- Recognized part filenames:
  - Exactly **1** `M_Head_<numericID>.fbx` (ID must exist in `Resources/character-master-20006200.json`)
  - Exactly **1** `M_Dress_*.fbx`
  - **0 or 1** `M_Hair_*.fbx`
  - **0 or more** `M_Acce_*.fbx`

## GUI Usage

Launch `Releases/scsp_mmd_tools/scsp_mmd_tools.exe` (or run `python studio_gui.py` from source) and use the top tab switcher to move between **角色 PMX** (Character PMX) and **动作 VMD** (Motion VMD), each keeping independent state while persisting settings to `studio_settings.json`:

### Character PMX Tab
1. Select the Blender executable and output root directory (top status badges live-check the detected Blender version plus built-in `ufbx FBX` and `PMX IO` readiness; the header also offers an optional **`保留 bust_up 形态键`** toggle, off by default).
2. Add one or more task folders to the queue:
   - **Single Build / Batch Build**: processes either the active item or the whole queue sequentially (continuing past individual failures); **Stop After Current** finishes the running model cleanly before stopping.
   - **Per-Task Overrides**: configure optional custom model names and multi-line **PMX comments** (blank by default — when blank, no forced template text is injected; supports `{name}`, `{romanized}`, `{id}`, and `{character_id}` placeholders, plus **Apply to Selected / Apply to All / Save as Default** persisted to `studio_settings.json`).
3. Each task writes into its own timestamped output directory without overwriting previous builds; `_work/` retains intermediate `.blend` scenes, config, reports, and logs. Keep the output `.pmx` and `textures/` folder together when loading in MMD.

### Motion VMD Tab
1. Select the **Song Task Folder** and **Output Directory** (**Default Target PMX** and **Custom Rules** are optional; leaving the default PMX blank exports against the built-in generic 45° A-pose MMD profile).
2. Click **Inspect Task** to list slots, character IDs, durations, and mouth sync files; optionally assign per-slot `.pmx` overrides or export all/selected slots (solo tracks without `_01` suffixes in `_exp.json` / `_root.unity3d` are handled automatically).
3. Click **Start Conversion** to generate per-slot `.vmd` files, shared-origin camera `*_camera.vmd`, `转换报告.json`, and `使用说明.txt`.

## Pipeline Stages & Key Features

Pipeline flow: `ufbx` FBX import & material/makeup prep → eye socket geometry & static eye highlight baking → unified skeleton, twist weight merging & 45° A-pose baking → body proportions, skin palette baking & custom skin Toons → cylindrical custom face normals → part merging → 62 MMD morphs → strict main-mesh PMX export → pure-Python `pmx_io` center controls, foot/toe IK, semi-standard bones & expression panels.

- **Materials & Per-Character Skin/Toon**: sets Blender roughness to `1` and PMX defaults `diffuse=(1,1,1)`, `specular=(0,0,0)`, `ambient=(0.25,0.25,0.25)`, `shininess=1`. Bakes `BodySkin` and isolated `M_face` palettes from character metadata along with `16×32` skin Toon ramps; visually verify final shading in your target MMD renderer.
- **Accessory Makeup & Helper Cleanup**: automatically transfers `M_face` makeup textures carried by accessory meshes onto the main head face material before hiding duplicate overlay cards, shadow proxies (`*_shadow`), `closeface`, placeholders, colliders, and `RoomShoes` (`Shoes` are kept).
- **Eyes & Eye Effects**: statically composites eye highlights while keeping source alpha (game dynamic UV shifts/jitter are intentionally not reproduced); rebuilds irises into alpha-trimmed concentric polar concave bowls (`32` radial segments × `4` rings), generates 1× egg-cup eye sockets via `eye_hole_{L,R}` loop mapping (with automatic `preserve_source` fallback + warnings only on malformed rims), and aligns eye bone pivots to spherical centers.
- **Facial Morphs (62 keys)**: strips broken/empty source keys (`口開け`, `左追加睫毛`, `右追加睫毛`, `tongue_up`), generates geometric eye expressions (`じと目`/`瞳小`/`瞳縦潰れ`/`びっくり`/`わぉ?!`/`なごみ`/`恐ろしい子！`), hidden in-head overlay effect cards (`星目`/`はぁと`/`はちゅ目`/`はぅ` with flat eyelid sealing), eyebrow keys (`個性`/`にこり`/`怒り`), and symmetric/full MMD mouth shapes (`ワ`/`にっこり`/`にやり`/`ω`/`ω□`/`▲`/`∧`/`□`/`ん`/`口角上げ`/`口角下げ`/`口横広げ`).
- **Body Proportions & Physics Status**: scales height/head/`bust_up` from character metadata alongside explicit `arm=0.95` and `hand=0.9` factors (`bust_up` is baked into rest geometry and custom corner normals). The main pipeline outputs a clean **physics-free** PMX. Experimental MMD native physics (`Tools/pe_physics.py`) currently does not meet quality expectations in MMD and is **marked as TODO / paused** (not recommended for production use).

## Running & Building from Source

Launching the GUI and running the **Character PMX** tab or `Tools/test_studio_pipeline.py` only require Python standard library (`>= 3.10`, with Tkinter for GUI) plus your local Blender installation. Running the **Motion VMD** converter (`motion_pipeline.py`), `tests/`, or building the full release requires [requirements.txt](requirements.txt) (`numpy` and `UnityPy`). On Windows, run [Setup-Env.cmd](Setup-Env.cmd) to automatically create `.venv` and install the motion conversion dependencies (pass `--mirror` for Tsinghua PyPI mirror, or `--build` to also install PyInstaller):

```powershell
# One-click motion environment setup: create .venv and install numpy / UnityPy
.\Setup-Env.cmd

# Launch GUI using the configured environment
.\.venv\Scripts\pythonw.exe studio_gui.py

# CLI: Character PMX
python studio_pipeline.py inspect <task_folder>
python studio_pipeline.py build <task_folder> --output <output_root> --blender <blender_exe>
python studio_pipeline.py build <task_folder> --output <output_root> --preserve-bust-morph
python studio_pipeline.py batch <task1> <task2> --output <output_root>

# CLI: Motion / Camera VMD (uses built-in pure-Python Humanoid solver by default)
python motion_pipeline.py inspect <song_task_folder>
python motion_pipeline.py build <song_task_folder> --output <output_root>

# Run unit tests
python -m unittest Tools/test_studio_pipeline.py -v
python -m unittest discover -s tests -v
```

### Packaging Standalone Executable (Windows)

Because `Releases/` binaries are not tracked in Git, you can build a portable directory (and optional release zip) locally with PyInstaller from a Python environment containing `requirements.txt` (`numpy`, `UnityPy`):

```powershell
powershell -ExecutionPolicy Bypass -File build_studio.ps1 -Python <python_exe> -CreateZip
```

Output is written to `Releases/scsp_mmd_tools/scsp_mmd_tools.exe` (preserving existing `Outputs/`). Prebuilt `Tools/ufbx_bridge.dll` is included in the repository; to recompile it from C source with MinGW `gcc`, run `Tools/native/build_ufbx.ps1`.

## Changelog

### v0.2.2 Preview (2026-11-23)

- Added optional **`保留 bust_up 形态键`** toggle (`--preserve-bust-morph`, off by default).

### v0.2.1 Preview (2026-11-19)

- Added approximate gaze toggle for motion conversion.
- Fixed `MaterialHolder` placeholder card cleanup and eyebrow card handling.

### v0.2.0 Preview (2026-11-18)

- **Integrated Motion / Camera VMD Conversion Tab**: added a top tab bar in `studio_gui.py` switching between `角色 PMX` and `动作 VMD` without altering the PMX processing pipeline; `MotionTab` persists `motion_*` paths in `studio_settings.json` and supports per-slot `.pmx` overrides or generic 45° A-pose export.
- **Built-in Pure-Python Humanoid Solver (External `PoseWorker` No Longer Required)**: added `BuiltinPoseBackend` backed by the pure mathematical kinematics table `Resources/humanoid_avatar.json` (`46 KB`, zero meshes or textures).
- **Non-linear Camera Timing, Solo Song Support & Lazy Dependency Loading**: added power-exponent `moveSpeed` sampling and early-plateau `easing` grid support, fallback for solo `_exp.json` files without `_01`, added `requirements.txt`, and lazy-loaded `motion_pipeline` so standard-library-only Python environments can still launch the Character PMX tab cleanly.

### v0.1.4 Preview (2026-11-17)

- Fixed compatibility for teeth and face accessories.

### v0.1.3 Preview (2026-11-17)

- Automatic removal of overlapping duplicate geometry.

### v0.1.2 Preview (2026-11-17)

- **Fixed `[WinError 87]` startup crash on Windows systems with unready/inaccessible drives**: `discover_default_blender()` in `studio_pipeline.py` now guards `is_dir()`, `glob()`, and `is_file()` against `OSError` (such as empty card readers/optical drives or disconnected network drives on `D:\`); `studio_gui.py` directory pickers include matching `OSError` guards and `Tools/test_studio_pipeline.py` adds regression coverage.

### v0.1.1 Preview (2026-11-17)

- **Fixed absolute texture paths in exported `.pmx` files**: `Tools/pe_core.py` (`finalize`) now writes portable relative paths (`textures\<filename>`) instead of full drive paths. This fixes two MikuMikuDance (MMD) bugs triggered when MMD joins the model folder with stored texture paths: **all-white materials** (broken texture lookups) and **crashes on load** when deep output directories pushed the combined path past Windows/MMD path length limits, and makes exported model folders portable across directories and PCs.
- **Updated audit & regression tests**: `Tools/audit_pe_result.py` now resolves texture paths relative to the output `.pmx` directory and flags any non-relative entries; added coverage in `Tools/test_studio_pipeline.py`.

### v0.1 Preview (2026-11-16)

- Initial preview release with built-in open-source `ufbx` FBX import and pure-Python `pmx_io` export for batch converting SCSP assets to clean, physics-free `.pmx`.

## Author's Note

This project was built with AI assistance, and parts of the pipeline rely on heuristic approximations rather than official specifications. No warranty of quality is provided for exported outputs; if you encounter any issues or bugs, please open an Issue or message me on social media.

## License & Disclaimer

- Source code in this repository is licensed under the **[GNU General Public License v3.0 (GPL-3.0)](LICENSE)**. Embedded C FBX parser [ufbx](ThirdParty/ufbx/LICENSE) and runtime dependency `UnityPy` are licensed under the MIT License; `numpy` is licensed under BSD-3-Clause; pure-Python PMX I/O, Humanoid solver, and all preprocessing scripts are independent open-source implementations.
- This repository contains **no** copyrighted third-party game models, textures, or audio/video assets and is intended strictly for personal non-commercial format research and interoperability study. Always respect original copyright holders' terms when handling character metadata, rigs, or textures.
