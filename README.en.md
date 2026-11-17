# scsp_mmd_tools (v0.1.1 Preview)

[简体中文](README.md) | **English**

Current release: **v0.1.1 Preview** — A desktop GUI and CLI batch pipeline that converts SCSP character assets into clean, physics-free MMD models (`.pmx` + `textures/`) cleanly in background Blender processes, **without requiring Unity, PmxEditor, or any commercial/external Blender add-ons**.

Keep `scsp_mmd_tools.exe` alongside `_internal/` when distributing or moving the packaged build.

## Requirements

- **Blender >= 4.1** installed locally (tested on **Blender 4.2 LTS** and **Blender 5.3.0**). Compatible Blender installations are auto-detected at startup (or can be specified via `SCSP_MMD_BLENDER` or browsed directly in the GUI).
- **Zero external plugin dependencies**: uses an embedded pure open-source **ufbx (MIT)** FBX loader (`Tools/ufbx_bridge.dll` + `Tools/ufbx_importer.py`) and a pure-Python **PMX 2.0/2.1 binary reader/writer** (`Tools/pmx_io.py`) instead of Better FBX, Blender `mmd_tools`, MMR, or Rigify.

## Input Folder Structure

Folders do **not** strictly have to come from the downloader (`job.json` and `manifest.json` are both optional). Manually arranged folders also work as long as they follow SCSP FBX naming, texture, and rig/topology conventions:

- Include a `models/` subfolder containing the `.fbx` parts and accompanying textures (or declare relative paths via `model_files` inside an optional `job.json`).
- Recognized part filenames:
  - Exactly **1** `M_Head_<numericID>.fbx` (ID must exist in `Resources/character-master-20006200.json`)
  - Exactly **1** `M_Dress_*.fbx`
  - **0 or 1** `M_Hair_*.fbx`
  - **0 or more** `M_Acce_*.fbx`

## GUI Usage

Launch `Releases/scsp_mmd_tools/scsp_mmd_tools.exe` (or run `python studio_gui.py` from source):

1. Select the Blender executable and output root directory (top status badges live-check the detected Blender version plus built-in `ufbx FBX` and `PMX IO` readiness).
2. Add one or more task folders to the queue:
   - **Single Build / Batch Build**: processes either the active item or the whole queue sequentially (continuing past individual failures); **Stop After Current** finishes the running model cleanly before stopping.
   - **Per-Task Overrides**: configure optional custom model names and multi-line **PMX comments** (blank by default — when blank, no forced template text is injected; supports `{name}`, `{romanized}`, `{id}`, and `{character_id}` placeholders, plus **Apply to Selected / Apply to All / Save as Default** persisted to `studio_settings.json`).
3. Each task writes into its own timestamped output directory without overwriting previous builds; `_work/` retains intermediate `.blend` scenes, config, reports, and logs. Keep the output `.pmx` and `textures/` folder together when loading in MMD.

## Pipeline Stages & Key Features

Pipeline flow: `ufbx` FBX import & material/makeup prep → eye socket geometry & static eye highlight baking → unified skeleton, twist weight merging & 45° A-pose baking → body proportions, skin palette baking & custom skin Toons → cylindrical custom face normals → part merging → 62 MMD morphs → strict main-mesh PMX export → pure-Python `pmx_io` center controls, foot/toe IK, semi-standard bones & expression panels.

- **Materials & Per-Character Skin/Toon**: sets Blender roughness to `1` and PMX defaults `diffuse=(1,1,1)`, `specular=(0,0,0)`, `ambient=(0.25,0.25,0.25)`, `shininess=1`. Bakes `BodySkin` and isolated `M_face` palettes from character metadata along with `16×32` skin Toon ramps; visually verify final shading in your target MMD renderer.
- **Accessory Makeup & Helper Cleanup**: automatically transfers `M_face` makeup textures carried by accessory meshes onto the main head face material before hiding duplicate overlay cards, shadow proxies (`*_shadow`), `closeface`, placeholders, colliders, and `RoomShoes` (`Shoes` are kept).
- **Eyes & Eye Effects**: statically composites eye highlights while keeping source alpha (game dynamic UV shifts/jitter are intentionally not reproduced); rebuilds irises into alpha-trimmed concentric polar concave bowls (`32` radial segments × `4` rings), generates 1× egg-cup eye sockets via `eye_hole_{L,R}` loop mapping (supporting non-strict topologies such as Kiriko, with automatic `preserve_source` fallback + warnings only on malformed rims), and aligns eye bone pivots to spherical centers.
- **Facial Morphs (62 keys)**: strips broken/empty source keys (`口開け`, `左追加睫毛`, `右追加睫毛`, `tongue_up`), generates geometric eye expressions (`じと目`/`瞳小`/`瞳縦潰れ`/`びっくり`/`わぉ?!`/`なごみ`/`恐ろしい子！`), hidden in-head overlay effect cards (`星目`/`はぁと`/`はちゅ目`/`はぅ` with flat eyelid sealing), eyebrow keys (`個性`/`にこり`/`怒り`), and symmetric/full MMD mouth shapes (`ワ`/`にっこり`/`にやり`/`ω`/`ω□`/`▲`/`∧`/`□`/`ん`/`口角上げ`/`口角下げ`/`口横広げ`).
- **Body Proportions & Optional Physics**: scales height/head/`bust_up` from character metadata alongside explicit `arm=0.95` and `hand=0.9` factors (`bust_up` is baked into rest geometry and custom corner normals). The main pipeline outputs a clean **physics-free** PMX; if you need MMD native rigid bodies & joints, run the standalone script:
  ```powershell
  python Tools/pe_physics.py --input <model.pmx> --output <model_physics.pmx>
  ```
  (also supports batch mode with `--input-dir` / `--output-dir`).

## Running & Building from Source

The launcher GUI and CLI depend solely on the Python standard library (`>= 3.10`, with Tkinter for GUI); all heavy mesh/rig processing executes inside your local Blender's bundled Python (no extra `pip` packages needed to run):

```powershell
# Launch GUI
python studio_gui.py

# CLI inspect / single build / batch build
python studio_pipeline.py inspect <task_folder>
python studio_pipeline.py build <task_folder> --output <output_root> --blender <blender_exe>
python studio_pipeline.py batch <task1> <task2> --output <output_root>

# Run core unit tests
python -m unittest Tools/test_studio_pipeline.py -v
```

### Packaging Standalone Executable (Windows)

Because `Releases/` binaries are not tracked in Git, you can build a portable directory (and optional release zip) locally with PyInstaller:

```powershell
powershell -ExecutionPolicy Bypass -File build_studio.ps1 -CreateZip
```

Output is written to `Releases/scsp_mmd_tools/scsp_mmd_tools.exe` (preserving existing `Outputs/`). Prebuilt `Tools/ufbx_bridge.dll` is included in the repository; to recompile it from C source with MinGW `gcc`, run `Tools/native/build_ufbx.ps1`.

## Changelog

### v0.1.1 Preview (2026-11-17)

- **Fixed absolute texture paths in exported `.pmx` files**: `Tools/pe_core.py` (`finalize`) now writes portable relative paths (`textures\<filename>`) instead of full drive paths. This fixes two MikuMikuDance (MMD) bugs triggered when MMD joins the model folder with stored texture paths: **all-white materials** (broken texture lookups) and **crashes on load** when deep output directories pushed the combined path past Windows/MMD path length limits, and makes exported model folders portable across directories and PCs.
- **Updated audit & regression tests**: `Tools/audit_pe_result.py` now resolves texture paths relative to the output `.pmx` directory and flags any non-relative entries; added coverage in `Tools/test_studio_pipeline.py`.

### v0.1 Preview (2026-11-16)

- Initial preview release with built-in open-source `ufbx` FBX import and pure-Python `pmx_io` export for batch converting SCSP assets to clean, physics-free `.pmx`.

## License & Disclaimer

- Source code in this repository is licensed under the **[GNU General Public License v3.0 (GPL-3.0)](LICENSE)**. Embedded C FBX parser [ufbx](ThirdParty/ufbx/LICENSE) is licensed under the MIT License; pure-Python PMX I/O and all preprocessing scripts are independent open-source implementations.
- This repository contains **no** copyrighted third-party game models, textures, or audio/video assets and is intended strictly for personal non-commercial format research and interoperability study. Always respect original copyright holders' terms when handling character metadata, rigs, or textures.
