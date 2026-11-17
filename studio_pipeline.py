"""Downloader task -> Blender preprocessing -> no-physics PMX, without Unity or PE."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
MIN_BLENDER_VERSION = (4, 1)
_BLENDER_VERSION_RE = re.compile(r'Blender\s+(\d+)\.(\d+)(?:\.(\d+))?')
_BLENDER_PROBE_CACHE: dict[tuple[str, int, int], dict[str, object]] = {}


def parse_blender_version(output: str) -> tuple[int, int, int] | None:
    match = _BLENDER_VERSION_RE.search(str(output or ''))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def probe_blender_executable(blender_path) -> dict[str, object]:
    """Check whether `blender_path` is a runnable Blender executable meeting MIN_BLENDER_VERSION."""
    try:
        path = Path(str(blender_path).strip().strip('"')).expanduser()
        if not path.is_file():
            return {'ok': False, 'supported': False, 'version': None, 'version_text': '', 'error': 'not a file'}
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        cache_key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
        cached = _BLENDER_PROBE_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)

        flags = 0x08000000 if os.name == 'nt' else 0
        proc = subprocess.run(
            [str(resolved), '--version'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=5,
            creationflags=flags,
        )
        version = parse_blender_version((proc.stdout or '') + '\n' + (proc.stderr or ''))
        supported = version is not None and version >= MIN_BLENDER_VERSION
        version_text = f'{version[0]}.{version[1]}.{version[2]}' if version else ''
        info: dict[str, object] = {
            'ok': proc.returncode == 0 and version is not None,
            'supported': supported,
            'version': version,
            'version_text': version_text,
            'error': None if supported else ('无法识别 Blender 版本' if version is None else f'需要 >= {MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]}'),
        }
        if info['ok']:
            _BLENDER_PROBE_CACHE[cache_key] = info
        return dict(info)
    except Exception as exc:
        return {'ok': False, 'supported': False, 'version': None, 'version_text': '', 'error': str(exc)}


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(root.glob(pattern)) if _safe_is_dir(root) else []
    except OSError:
        return []


def discover_default_blender() -> Path:
    """Return SCSP_MMD_BLENDER if set, else the newest installed compatible Blender (>= 4.1)."""
    env_value = os.environ.get('SCSP_MMD_BLENDER', '').strip()
    if env_value:
        return Path(env_value)

    candidates: list[Path] = []
    which_hit = shutil.which('blender')
    if which_hit:
        candidates.append(Path(which_hit))
    for pattern_root, subglob in (
        (Path('C:/'), 'blender*/blender.exe'),
        (Path('D:/'), 'blender*/blender.exe'),
        (Path('C:/Program Files/Blender Foundation'), '*/blender.exe'),
        (Path('D:/Program Files/Blender Foundation'), '*/blender.exe'),
    ):
        candidates.extend(_safe_glob(pattern_root, subglob))
    candidates.append(Path('D:/blender-5.3.0/blender.exe'))

    seen: set[str] = set()
    best: tuple[tuple[int, int, int], Path] | None = None
    for cand in candidates:
        try:
            key = str(cand.resolve(strict=False)).lower()
        except Exception:
            key = str(cand).lower()
        if key in seen or not _safe_is_file(cand):
            continue
        seen.add(key)
        probe = probe_blender_executable(cand)
        version = probe.get('version')
        if probe.get('supported') and isinstance(version, tuple):
            if best is None or version > best[0]:
                best = (version, cand)
    if best is not None:
        return best[1]
    return Path(which_hit) if which_hit else Path('blender')


def validate_blender_executable(blender_path) -> Path:
    blender = Path(blender_path).resolve(strict=False)
    if not blender.is_file():
        raise ValueError(f'Blender 可执行文件不存在：{blender}')
    probe = probe_blender_executable(blender)
    version = probe.get('version')
    if not probe.get('ok') or version is None:
        raise ValueError(f'无法运行或识别 Blender 版本：{blender}')
    if not probe.get('supported'):
        raise ValueError(
            f'当前 Blender 版本为 {probe.get("version_text")}，需要 Blender '
            f'{MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]} 或更高版本'
        )
    return blender.resolve(strict=True)


DEFAULT_BLENDER = discover_default_blender()
TODO = []


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_task(task_dir):
    task = Path(task_dir).resolve(strict=True)
    if not task.is_dir():
        raise ValueError('请选择下载任务文件夹')
    job = read_json(task / 'job.json') if (task / 'job.json').exists() else {}
    if job.get('status') not in (None, 'completed'):
        raise ValueError('下载任务未完成：' + str(job.get('status')))
    if job.get('dependency_complete') is False or job.get('missing_count', 0):
        raise ValueError('下载任务缺少依赖文件，请先完成下载')
    declared = job.get('model_files')
    if declared is not None and not isinstance(declared, list):
        raise ValueError('job.model_files 必须是数组')
    files = [task / x for x in declared] if declared is not None else list((task / 'models').rglob('*.fbx'))
    parts = []
    seen = set()
    for source in files:
        source = source.resolve(strict=True)
        if not source.is_relative_to(task):
            raise ValueError('模型路径超出任务目录：' + str(source))
        if source in seen:
            continue
        seen.add(source)
        match = re.fullmatch(r'M_(Head|Dress|Hair|Acce)_(.+)\.fbx', source.name, re.I)
        if not match:
            raise ValueError('无法识别模型部件：' + source.name)
        kind = match[1].lower()
        if not source.is_file():
            raise ValueError('模型不是文件：' + str(source))
        parts.append({'kind': kind, 'path': str(source)})
    for kind in ('head', 'dress'):
        if sum(p['kind'] == kind for p in parts) != 1:
            raise ValueError('每个任务必须恰有一个 ' + kind + ' 模型')
    if sum(p['kind'] == 'hair' for p in parts) > 1:
        raise ValueError('任务含多个发型，请拆分为不同下载任务')
    parts.sort(key=lambda p: (['head', 'dress', 'hair', 'acce'].index(p['kind']), p['path'].casefold()))
    head = next(p for p in parts if p['kind'] == 'head')
    head_match = re.fullmatch(r'M_Head_(\d+)\.fbx', Path(head['path']).name, re.I)
    if head_match is None:
        raise ValueError('头部文件名缺少数字角色 ID：' + head['path'])
    head_id = int(head_match[1])
    snapshot = read_json(ROOT / 'Resources/character-master-20006200.json')
    record = next((r for r in snapshot['records'] if r['id'] == head_id), None)
    if record is None:
        raise ValueError(f'本地角色数据不包含 head {head_id}，不能自动套用其他角色比例')
    roman = (record['lastNameRoma'] + ' ' + record['firstNameRoma']).title()
    names = read_json(ROOT / 'Resources/character-names.json')['characters']
    warnings = []
    manifest = read_json(task / 'manifest.json') if (task / 'manifest.json').exists() else {}
    resource = str(manifest.get('resource_version', manifest.get('resourceVersion', '')))
    if resource and resource.split('@')[0] != snapshot['resourceVersion'].split('@')[0]:
        warnings.append('任务资源版本与冻结角色资料版本不同；使用本工具记录的比例配方')
    return {'task_dir': str(task), 'name': names.get(f'{head_id:02}', roman),
            'romanized_name': roman, 'character_id': head_id, 'head_id': head_id,
            'record': record, 'parts': parts, 'warnings': warnings,
            'resource_version': resource}


def _safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:72]
    if not name:
        raise ValueError('角色名称不能为空')
    if name.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}:
        name = '_' + name
    return name


def format_pmx_comment(comment, info=None):
    """Return user-provided PMX comment text verbatim with optional placeholder substitution."""
    text = str(comment or '').replace('\r\n', '\n').replace('\r', '\n').strip('\n')
    if not text.strip():
        return ''
    if info:
        char_id = info.get('character_id', '')
        char_id_text = '' if char_id is None else str(char_id)
        replacements = {
            '{name}': str(info.get('name', '') or ''),
            '{romanized}': str(info.get('romanized_name', '') or ''),
            '{id}': char_id_text,
            '{character_id}': char_id_text,
        }
        for token, value in replacements.items():
            text = text.replace(token, value)
    return text.rstrip()


def build_task(
    task_dir,
    output_root,
    blender_path=DEFAULT_BLENDER,
    name_override=None,
    comment='',
    progress=None,
):
    info = inspect_task(task_dir)
    blender = validate_blender_executable(blender_path)
    name = str(name_override or info['romanized_name']).strip()
    label = _safe_filename(name)
    stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    out = Path(output_root).resolve() / (label + '_' + Path(task_dir).name + '_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    work = out / '_work'
    work.mkdir()
    logs = work / 'logs'
    logs.mkdir()
    formatted_comment = format_pmx_comment(comment, info)
    result = {'schema': 1, 'status': 'running', 'input': info, 'output_dir': str(out),
              'name': name, 'comment': formatted_comment, 'pending': TODO, 'stages': [],
              'effect_validation': '由用户手动检验；未进行动作效果验收',
              'skin_note': '按角色主数据身体/面部明暗Hex色板烘培肤色贴图并生成专属皮肤Toon渐变；贴图乘扩散色111、反射色000、环境色0.25、高光指数1',
              'proportion_note': '角色主数据height/head/bust + 显式arm=0.95 hand=0.9配方，未恢复游戏原生骨骼公式'}
    def emit(percent, message):
        if progress:
            progress(percent=percent, message=message)
    def stage(script, arguments, percent, title):
        emit(percent, title)
        log = logs / (f'{len(result["stages"]) + 1:02}-' + script.replace('.py', '') + '.log')
        cmd = [str(blender), '--background', '--factory-startup', '--disable-autoexec',
               '--python-exit-code', '1', '--python', str(ROOT / 'Tools' / script), '--', *map(str, arguments)]
        start = time.monotonic()
        with log.open('wb') as stream:
            # Below-normal priority keeps background Blender builds from starving GUI drag/resize.
            flags = (0x08000000 | 0x00004000) if os.name == 'nt' else 0
            process = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=flags)
        result['stages'].append({'script': script, 'log': str(log), 'seconds': round(time.monotonic()-start, 2), 'exit_code': process.returncode})
        write_json(out / '制作报告.json', result)
        if process.returncode:
            tail = log.read_text(encoding='utf-8', errors='replace')[-5000:]
            raise RuntimeError(f'{title}失败，日志：{log}\n{tail}')
    try:
        config = read_json(ROOT / 'Resources/pipeline-defaults.json')
        master = ROOT / 'Resources/character-master-20006200.json'
        record = info['record']
        config.update({'sample': Path(task_dir).name, 'parts': info['parts'], 'record': record,
                       'master_snapshot': str(master), 'master_snapshot_sha256': sha(master),
                       'parameters': {'height': record['heightScale'], 'head': record['headShape'],
                                      'bust': record['shape'], 'arm': .95, 'hand': .9},
                       'eye_texture_output_dir': str(work / 'eye-textures'),
                       'skin_texture_output_dir': str(work / 'skin-textures'),
                       'render_preview': False})
        tools_dir = ROOT / 'Tools'
        for dep in ('ufbx_bridge.dll', 'ufbx_importer.py', 'pmx_io.py'):
            if not (tools_dir / dep).is_file():
                raise FileNotFoundError(f'缺少内置开源 IO 组件：{tools_dir / dep}')
        cfg = work / 'config.json'
        write_json(cfg, config)
        imported = work / 'import'
        prepared = work / 'prepared'
        stage('blender_preprocess.py', ['--config', cfg, '--output', imported, '--import-only'], 3, '导入 FBX、整理部件和材质')
        stage('modular_preprocess.py', ['--input-blend', imported/'01-import.blend', '--import-report', imported/'report.json', '--config', cfg, '--output', prepared], 22, '眼部、骨架、比例、法向和网格处理')
        prep_report = read_json(prepared/'report.json')
        result['warnings'] = (
            info['warnings']
            + prep_report['modules'].get('eye_texture', {}).get('warnings', [])
            + prep_report['modules'].get('skin_processing', {}).get('warnings', [])
            + prep_report['modules'].get('eye_processing', {}).get('warnings', [])
        )
        for warning in result['warnings']:
            emit(53, '提醒：' + str(warning))
        base = work/'00-base.pmx'
        stage('studio_export.py', ['--input', prepared/'preprocessed.blend', '--output', base, '--name', name, '--comment', formatted_comment, '--report', work/'export-report.json'], 55, '导出基础 PMX 与材质')
        body, leg, semi = [work/p for p in ('01-body.pmx', '02-leg.pmx', '03-semi.pmx')]
        stage('pe_core.py', ['--input', base, '--output', body, '--phase', 'body'], 68, '生成中心控制层')
        stage('pe_controls.py', ['--input', body, '--output', leg, '--phase', 'leg'], 75, '生成足与脚趾 IK')
        stage('pe_controls.py', ['--input', leg, '--output', semi, '--phase', 'semi'], 82, '生成半标准控制骨')
        final = out / (label + '.pmx')
        stage('pe_core.py', ['--input', semi, '--output', final, '--phase', 'finalize'], 90, '整理表情面板与贴图目录')
        stage('audit_pe_result.py', ['--input', base, '--output', final, '--report', work/'structural-audit.json'], 96, '检查 PMX 数据与纹理完整性')
        result.update(status='completed', pmx=str(final), sha256=sha(final),
                      preprocessing_report=str(prepared/'report.json'), export_report=str(work/'export-report.json'),
                      textures=[{'path': p.relative_to(out).as_posix(), 'sha256': sha(p)} for p in sorted((out/'textures').iterdir())])
        todo_note = ('\n\n待做：\n' + '\n'.join(TODO)) if TODO else ''
        (out / '使用说明.txt').write_text('scsp_mmd_tools\n\n输出：PMX + textures贴图目录，请一同移动。\n不依赖Unity或PmxEditor；无物理。\n_work保留制作阶段、日志及可手动检验的Blender工程。'+todo_note+'\n\n'+result['skin_note']+'\n'+result['proportion_note']+'\n眼睛高光为静态合成，未恢复游戏高光位移/动画。\n', encoding='utf-8-sig')
        write_json(out/'制作报告.json', result)
        emit(100, '制作完成：' + str(final))
        return result
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        write_json(out/'制作报告.json', result)
        raise


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'build', 'batch'])
    parser.add_argument('tasks', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path.cwd()/'Outputs')
    parser.add_argument('--blender', type=Path, default=DEFAULT_BLENDER)
    parser.add_argument('--name-override', default=None)
    parser.add_argument('--comment', default='')
    args = parser.parse_args()
    results = []
    for task in args.tasks:
        try:
            value = (
                inspect_task(task)
                if args.action == 'inspect'
                else build_task(
                    task,
                    args.output,
                    args.blender,
                    name_override=args.name_override,
                    comment=args.comment,
                    progress=lambda **x: print(json.dumps(x, ensure_ascii=False), flush=True),
                )
            )
            results.append({'task': str(task), 'status': 'completed', 'result': value})
        except Exception as exc:
            results.append({'task': str(task), 'status': 'failed', 'error': str(exc)})
            if args.action != 'batch':
                break
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any(r['status'] == 'failed' for r in results))

if __name__ == '__main__':
    raise SystemExit(main())
