from __future__ import annotations

import ctypes
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from studio_pipeline import (
    DEFAULT_BLENDER as _PIPELINE_DEFAULT_BLENDER,
    MIN_BLENDER_VERSION,
    probe_blender_executable,
    validate_blender_executable,
)

DEFAULT_BLENDER = os.environ.get(
    "SCSP_MMD_BLENDER",
    str(_PIPELINE_DEFAULT_BLENDER),
)
DEFAULT_OUTPUT_ROOT = APP_DIR / "Outputs" / "studio"
STUDIO_SETTINGS_PATH = APP_DIR / "studio_settings.json"
EVENT_POLL_MS = 100
EVENT_POLL_ACTIVE_MS = 80
EVENT_POLL_IDLE_MS = 250


def load_studio_settings(path: Path | None = None) -> dict[str, Any]:
    """Load persisted Studio GUI settings from JSON, returning {} if unavailable."""
    target = STUDIO_SETTINGS_PATH if path is None else path
    try:
        if target.is_file():
            data = json.loads(target.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_studio_settings(
    data: dict[str, Any], path: Path | None = None
) -> None:
    """Persist Studio GUI settings atomically via a sibling temp file."""
    target = STUDIO_SETTINGS_PATH if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, target)


def enable_windows_dpi_awareness() -> None:
    """Enable Per-Monitor v2 DPI awareness on Windows before Tk initializes."""
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class StudioTask:
    key: str
    task_dir: Path
    status: str = "识别中"
    progress: int = 0
    name: str = ""
    romanized_name: str = ""
    character_id: str = ""
    head_id: str = ""
    parts: list[str] = field(default_factory=list)
    override_name: str = ""
    comment: str = ""
    output_dir: Path | None = None
    pmx: Path | None = None
    error: str = ""
    detail: str = ""

    @property
    def display_name(self) -> str:
        return self.override_name or self.name or self.task_dir.name

    @property
    def parts_display(self) -> str:
        return " / ".join(self.parts) if self.parts else "—"


@dataclass(frozen=True)
class BuildJob:
    task_key: str
    task_dir: Path
    name_override: str
    comment: str


def _single_line(value: object, fallback: str = "—") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).replace("\r", "\n").splitlines()).strip()
    return text or fallback


def _parts_from_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        raw_parts = list(value)
    else:
        raw_parts = [value]
    result: list[str] = []
    for part in raw_parts:
        if isinstance(part, dict):
            label = (
                part.get("name")
                or part.get("part_name")
                or part.get("id")
                or part.get("source")
                or part.get("kind")
            )
        else:
            label = part
        text = _single_line(label, "")
        if text and text not in result:
            result.append(text)
    return result


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _absolute_path(raw: str, label: str) -> Path:
    cleaned = raw.strip().strip('"')
    if not cleaned:
        raise ValueError(f"请填写{label}。")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        raise ValueError(f"{label}必须是绝对路径：{cleaned}")
    return path.resolve(strict=False)


def _friendly_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _pick_multiple_directories_win32(
    parent_hwnd: int,
    title: str,
    initial_dir: Path,
) -> list[str] | None:
    """Open native Windows Explorer folder picker with Ctrl/Shift multi-select enabled."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import POINTER, WINFUNCTYPE, byref, c_long, c_ubyte, c_uint32, c_ulong, c_void_p, c_wchar_p, wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", c_ubyte * 8),
        ]

        @classmethod
        def from_str(cls, text: str) -> "_GUID":
            return cls.from_buffer_copy(uuid.UUID(text).bytes_le)

    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32
    ole32.CoInitialize(None)
    clsid_fod = _GUID.from_str("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
    iid_fod = _GUID.from_str("{D57C7288-D4AD-4768-BE02-9D969532D960}")
    iid_shell_item = _GUID.from_str("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")
    dlg = c_void_p()
    hr = ole32.CoCreateInstance(byref(clsid_fod), None, 1, byref(iid_fod), byref(dlg))
    if hr != 0 or not dlg:
        return None

    def _release(ptr: c_void_p) -> None:
        if ptr:
            vt = ctypes.cast(ctypes.cast(ptr, POINTER(c_void_p))[0], POINTER(c_void_p))
            WINFUNCTYPE(c_ulong, c_void_p)(vt[2])(ptr)

    try:
        vtbl = ctypes.cast(ctypes.cast(dlg, POINTER(c_void_p))[0], POINTER(c_void_p))
        show = WINFUNCTYPE(c_long, c_void_p, c_void_p)(vtbl[3])
        set_options = WINFUNCTYPE(c_long, c_void_p, c_uint32)(vtbl[9])
        get_options = WINFUNCTYPE(c_long, c_void_p, POINTER(c_uint32))(vtbl[10])
        set_folder = WINFUNCTYPE(c_long, c_void_p, c_void_p)(vtbl[12])
        set_title = WINFUNCTYPE(c_long, c_void_p, c_wchar_p)(vtbl[17])
        get_results = WINFUNCTYPE(c_long, c_void_p, POINTER(c_void_p))(vtbl[27])

        opts = c_uint32()
        if get_options(dlg, byref(opts)) == 0:
            # FOS_PICKFOLDERS (0x20) | FOS_FORCEFILESYSTEM (0x40) | FOS_ALLOWMULTISELECT (0x200) | FOS_PATHMUSTEXIST (0x800)
            set_options(dlg, opts.value | 0x20 | 0x40 | 0x200 | 0x800)
        set_title(dlg, title)

        if initial_dir.is_dir():
            folder_item = c_void_p()
            shell32.SHCreateItemFromParsingName.argtypes = [c_wchar_p, c_void_p, POINTER(_GUID), POINTER(c_void_p)]
            if shell32.SHCreateItemFromParsingName(str(initial_dir), None, byref(iid_shell_item), byref(folder_item)) == 0:
                set_folder(dlg, folder_item)
                _release(folder_item)

        hr_show = show(dlg, c_void_p(parent_hwnd) if parent_hwnd else None)
        if (hr_show & 0xFFFFFFFF) == 0x800704C7:
            return []
        if hr_show != 0:
            return None

        results_arr = c_void_p()
        if get_results(dlg, byref(results_arr)) != 0 or not results_arr:
            return []
        try:
            arr_vtbl = ctypes.cast(ctypes.cast(results_arr, POINTER(c_void_p))[0], POINTER(c_void_p))
            get_count = WINFUNCTYPE(c_long, c_void_p, POINTER(c_uint32))(arr_vtbl[7])
            get_item = WINFUNCTYPE(c_long, c_void_p, c_uint32, POINTER(c_void_p))(arr_vtbl[8])
            count = c_uint32()
            if get_count(results_arr, byref(count)) != 0:
                return []
            paths: list[str] = []
            for idx in range(count.value):
                item = c_void_p()
                if get_item(results_arr, idx, byref(item)) != 0 or not item:
                    continue
                try:
                    item_vtbl = ctypes.cast(ctypes.cast(item, POINTER(c_void_p))[0], POINTER(c_void_p))
                    get_name = WINFUNCTYPE(c_long, c_void_p, c_uint32, POINTER(c_void_p))(item_vtbl[5])
                    psz = c_void_p()
                    # SIGDN_FILESYSPATH = 0x80058000
                    if get_name(item, 0x80058000, byref(psz)) == 0 and psz:
                        try:
                            text = ctypes.wstring_at(psz.value)
                            if text:
                                paths.append(text)
                        finally:
                            ole32.CoTaskMemFree(psz)
                finally:
                    _release(item)
            return paths
        finally:
            _release(results_arr)
    finally:
        _release(dlg)


def _is_direct_task_dir(path: Path) -> bool:
    return (path / "models").is_dir() or any(path.glob("*.fbx"))


def _expand_task_directories(raw_paths: list[str]) -> list[str]:
    """Expand any selected parent folder (e.g. web-downloads) into direct task folders."""
    expanded: list[str] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_dir() or _is_direct_task_dir(path):
            expanded.append(raw)
            continue
        children = sorted(
            child for child in path.iterdir()
            if child.is_dir() and _is_direct_task_dir(child)
        )
        if children:
            expanded.extend(str(child) for child in children)
        else:
            expanded.append(raw)
    return expanded


class StudioGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._tasks: dict[str, StudioTask] = {}
        self._selected_key = ""
        self._building = False
        self._closing = False
        self._stop_event: threading.Event | None = None
        self._pipeline: Any = None
        self._pipeline_lock = threading.Lock()
        self._events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._inspect_jobs: queue.Queue[tuple[str, Path] | None] = queue.Queue()
        self._syncing_details = False
        self._log_lines = 0
        self._log_visible = True
        self._sash_initialized = False
        self._last_configure_at = 0.0
        self._settings = load_studio_settings()
        self._default_comment = str(
            self._settings.get("default_pmx_comment", "") or ""
        )

        self.blender_var = tk.StringVar(value=DEFAULT_BLENDER)
        self.downloader_var = tk.StringVar()
        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.name_override_var = tk.StringVar()
        self.comment_label_var = tk.StringVar(value="默认 PMX 说明")
        self.status_var = tk.StringVar(value="就绪")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.queue_count_var = tk.StringVar(value="0 项")
        self.detail_title_var = tk.StringVar(value="尚未选择任务")
        self.detail_vars = {
            "name": tk.StringVar(value="—"),
            "romanized": tk.StringVar(value="—"),
            "character": tk.StringVar(value="—"),
            "head": tk.StringVar(value="—"),
            "parts": tk.StringVar(value="—"),
            "output": tk.StringVar(value="—"),
        }

        self._configure_window()
        self._configure_style()
        self._build_layout()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(EVENT_POLL_ACTIVE_MS, self._poll_events)
        self._refresh_env_badges()

        self._inspector_thread = threading.Thread(
            target=self._inspector_loop,
            name="scsp-inspector",
            daemon=True,
        )
        self._inspector_thread.start()
        self._append_log("界面已就绪。添加任务文件夹后会读取角色识别信息。")
        self._show_details_for_selection()
        self._refresh_actions()

    def _configure_window(self) -> None:
        self.root.title("scsp_mmd_tools v0.1 (Preview)")
        try:
            dpi = float(self.root.winfo_fpixels("1i"))
        except Exception:
            dpi = 96.0
        scale = max(1.0, min(2.5, dpi / 96.0))
        try:
            self.root.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass
        width = int(round(1220 * scale))
        height = int(round(860 * scale))
        min_w = int(round(980 * scale))
        min_h = int(round(720 * scale))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min_w, min_h)
        self.root.configure(bg="#F5F5F7")

    def _configure_style(self) -> None:
        ui_font = ("Microsoft YaHei UI", 9)
        self.root.option_add("*Font", ui_font)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        canvas_bg = "#F5F5F7"
        card_bg = "#FFFFFF"
        sub_bg = "#F9F9FB"
        border = "#D1D1D6"
        hairline = "#E5E5EA"
        text_primary = "#1D1D1F"
        text_secondary = "#6E6E73"
        accent = "#007AFF"
        accent_hover = "#0062CC"
        accent_disabled = "#A8CEFF"
        danger = "#FF3B30"
        danger_hover = "#D72D23"
        danger_disabled = "#F4B8B4"

        style.configure("TFrame", background=canvas_bg)
        style.configure("Card.TFrame", background=card_bg)
        style.configure("Sub.TFrame", background=sub_bg)
        style.configure("TLabel", background=card_bg, foreground=text_primary)
        style.configure(
            "Section.TLabel",
            background=card_bg,
            foreground=text_primary,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=card_bg,
            foreground=text_secondary,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Hint.TLabel",
            background=card_bg,
            foreground="#86868B",
            font=("Microsoft YaHei UI", 8),
        )
        style.configure(
            "Card.TCheckbutton",
            background=card_bg,
            foreground=text_secondary,
            font=("Microsoft YaHei UI", 8),
        )
        style.configure(
            "AppTitle.TLabel",
            background=card_bg,
            foreground=text_primary,
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        style.configure(
            "AppSubtitle.TLabel",
            background=card_bg,
            foreground=text_secondary,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Badge.TLabel",
            background="#F2F2F7",
            foreground=text_secondary,
            font=("Microsoft YaHei UI", 8, "bold"),
            padding=(8, 2),
        )
        style.configure(
            "Apple.TEntry",
            fieldbackground=sub_bg,
            foreground=text_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            insertcolor=text_primary,
            padding=(8, 5),
        )
        style.map(
            "Apple.TEntry",
            bordercolor=[("focus", accent), ("disabled", hairline)],
            lightcolor=[("focus", accent)],
            darkcolor=[("focus", accent)],
            fieldbackground=[("disabled", "#F2F2F7")],
        )
        style.configure(
            "TButton",
            background="#FFFFFF",
            foreground=text_primary,
            bordercolor=border,
            lightcolor="#FFFFFF",
            darkcolor=border,
            relief="flat",
            padding=(12, 5),
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "TButton",
            background=[("active", "#F2F2F7"), ("disabled", "#F5F5F7")],
            foreground=[("disabled", "#AEAEB2")],
            bordercolor=[("active", "#C7C7CC"), ("disabled", hairline)],
        )
        style.configure(
            "Soft.TButton",
            background="#F2F2F7",
            foreground=text_primary,
            bordercolor="#F2F2F7",
            lightcolor="#F2F2F7",
            darkcolor="#F2F2F7",
            relief="flat",
            padding=(11, 5),
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "Soft.TButton",
            background=[("active", "#E5E5EA"), ("disabled", "#F5F5F7")],
            foreground=[("disabled", "#AEAEB2")],
        )
        style.configure(
            "Primary.TButton",
            background=accent,
            foreground="#FFFFFF",
            bordercolor=accent,
            lightcolor=accent,
            darkcolor=accent,
            relief="flat",
            padding=(13, 5),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", accent_hover), ("disabled", accent_disabled)],
            bordercolor=[("active", accent_hover), ("disabled", accent_disabled)],
            foreground=[("disabled", "#FFFFFF")],
        )
        style.configure(
            "Stop.TButton",
            background=danger,
            foreground="#FFFFFF",
            bordercolor=danger,
            lightcolor=danger,
            darkcolor=danger,
            relief="flat",
            padding=(12, 5),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Stop.TButton",
            background=[("active", danger_hover), ("disabled", danger_disabled)],
            bordercolor=[("active", danger_hover), ("disabled", danger_disabled)],
            foreground=[("disabled", "#FFFFFF")],
        )
        style.configure(
            "Treeview",
            rowheight=32,
            background=card_bg,
            fieldbackground=card_bg,
            foreground=text_primary,
            borderwidth=0,
            relief="flat",
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", "#E8F2FF")],
            foreground=[("selected", "#1D1D1F")],
        )
        style.configure(
            "Treeview.Heading",
            background=sub_bg,
            foreground=text_secondary,
            bordercolor=hairline,
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(8, 7),
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#EFEFF4")],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=hairline,
            background=accent,
            borderwidth=0,
            thickness=7,
        )
        style.configure(
            "TPanedwindow",
            background=canvas_bg,
            sashrelief="flat",
            sashpad=3,
        )

    @staticmethod
    def _create_card(parent: tk.Misc) -> ttk.Frame:
        outer = tk.Frame(parent, bg="#E5E5EA", padx=1, pady=1)
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="both", expand=True)
        inner._card_outer = outer  # type: ignore[attr-defined]
        return inner

    def _build_layout(self) -> None:
        header_wrap = tk.Frame(self.root, bg="#FFFFFF")
        header_wrap.pack(fill="x")
        banner = tk.Frame(header_wrap, bg="#FFFFFF", padx=20, pady=12)
        banner.pack(fill="x")
        tk.Frame(header_wrap, bg="#E5E5EA", height=1).pack(fill="x")

        title_row = tk.Frame(banner, bg="#FFFFFF")
        title_row.pack(side="left")
        logo = tk.Label(
            title_row,
            text="PMX",
            bg="#007AFF",
            fg="#FFFFFF",
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=9,
            pady=4,
        )
        logo.pack(side="left", padx=(0, 12))
        title_stack = tk.Frame(title_row, bg="#FFFFFF")
        title_stack.pack(side="left")
        ttk.Label(
            title_stack,
            text="scsp_mmd_tools",
            style="AppTitle.TLabel",
        ).pack(anchor="w")

        header_right = tk.Frame(banner, bg="#FFFFFF")
        header_right.pack(side="right")
        self.env_badge_frame = tk.Frame(header_right, bg="#FFFFFF")
        self.env_badge_frame.pack(side="left", padx=(0, 12))
        self.open_button = ttk.Button(
            header_right,
            text="打开输出目录",
            style="Soft.TButton",
            command=self._open_output_directory,
        )
        self.open_button.pack(side="left")

        paths_card = self._create_card(self.root)
        paths_card._card_outer.pack(fill="x", padx=18, pady=(14, 10))  # type: ignore[attr-defined]
        paths_inner = ttk.Frame(paths_card, style="Card.TFrame", padding=(16, 12))
        paths_inner.pack(fill="x")
        paths_inner.grid_columnconfigure(1, weight=1)

        paths_head = ttk.Frame(paths_inner, style="Card.TFrame")
        paths_head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(
            paths_head,
            text="工作区与路径配置",
            style="Section.TLabel",
        ).pack(side="left")

        self.path_entries: list[ttk.Entry] = []
        self.path_buttons: list[ttk.Button] = []
        self.blender_entry = self._add_path_row(
            paths_inner,
            1,
            "Blender 可执行文件",
            self.blender_var,
            self._browse_blender,
        )
        self.downloader_entry = self._add_path_row(
            paths_inner,
            2,
            "下载器任务目录",
            self.downloader_var,
            self._browse_downloader,
        )
        self.output_entry = self._add_path_row(
            paths_inner,
            3,
            "PMX 输出根目录",
            self.output_root_var,
            self._browse_output_root,
        )
        self.path_entries.extend(
            [self.blender_entry, self.downloader_entry, self.output_entry]
        )

        body = tk.PanedWindow(
            self.root,
            orient="horizontal",
            sashwidth=7,
            sashpad=1,
            sashrelief="flat",
            bg="#F5F5F7",
            borderwidth=0,
            relief="flat",
            opaqueresize=False,
            showhandle=False,
        )

        bottom_card = self._create_card(self.root)
        bottom_card._card_outer.pack(side="bottom", fill="x", padx=18, pady=(0, 14))  # type: ignore[attr-defined]
        body.pack(side="top", fill="both", expand=True, padx=18, pady=(0, 10))

        tasks_card = self._create_card(body)
        details_card = self._create_card(body)
        body.add(tasks_card._card_outer, stretch="always", minsize=460)  # type: ignore[attr-defined]
        body.add(details_card._card_outer, stretch="always", minsize=300)  # type: ignore[attr-defined]
        self._body_paned = body
        body.bind("<Map>", self._on_body_mapped, add="+")

        tasks_pad = ttk.Frame(tasks_card, style="Card.TFrame", padding=(14, 12))
        tasks_pad.pack(fill="both", expand=True)

        tasks_header = ttk.Frame(tasks_pad, style="Card.TFrame")
        tasks_header.pack(fill="x", pady=(0, 10))
        ttk.Label(
            tasks_header,
            text="制作队列",
            style="Section.TLabel",
        ).pack(side="left")
        ttk.Label(
            tasks_header,
            textvariable=self.queue_count_var,
            style="Badge.TLabel",
        ).pack(side="left", padx=(8, 0))

        toolbar = ttk.Frame(tasks_pad, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 10))
        left_actions = ttk.Frame(toolbar, style="Card.TFrame")
        left_actions.pack(side="left")
        right_actions = ttk.Frame(toolbar, style="Card.TFrame")
        right_actions.pack(side="right")

        self.add_button = ttk.Button(
            left_actions,
            text="+ 添加任务",
            style="Soft.TButton",
            command=lambda: self._choose_tasks(multiple=False),
        )
        self.add_many_button = ttk.Button(
            left_actions,
            text="批量导入…",
            style="Soft.TButton",
            command=lambda: self._choose_tasks(multiple=True),
        )
        self.remove_button = ttk.Button(
            left_actions,
            text="移除",
            style="TButton",
            command=self._remove_selected,
        )
        for btn in (self.add_button, self.add_many_button, self.remove_button):
            btn.pack(side="left", padx=(0, 6))

        self.single_button = ttk.Button(
            right_actions,
            text="制作所选",
            style="Soft.TButton",
            command=self._build_single,
        )
        self.batch_button = ttk.Button(
            right_actions,
            text="开始批量",
            style="Primary.TButton",
            command=self._build_batch,
        )
        self.stop_button = ttk.Button(
            right_actions,
            text="完成后停止",
            style="Stop.TButton",
            command=self._request_stop_after_current,
        )
        self.single_button.pack(side="left", padx=(0, 6))
        self.batch_button.pack(side="left", padx=(0, 6))
        self.stop_button.pack(side="left")

        tree_shell = tk.Frame(tasks_pad, bg="#E5E5EA", padx=1, pady=1)
        tree_shell.pack(fill="both", expand=True)
        tree_frame = ttk.Frame(tree_shell, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)

        columns = (
            "status",
            "name",
            "romanized",
            "character",
            "head",
            "parts",
            "progress",
            "path",
        )
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=6,
        )
        headings = {
            "status": "状态",
            "name": "角色名",
            "romanized": "罗马字",
            "character": "角色 ID",
            "head": "头部 ID",
            "parts": "部件",
            "progress": "进度",
            "path": "任务路径",
        }
        widths = {
            "status": 68,
            "name": 100,
            "romanized": 88,
            "character": 82,
            "head": 82,
            "parts": 110,
            "progress": 52,
            "path": 160,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column], anchor="w")
            self.tree.column(
                column,
                width=widths[column],
                minwidth=48,
                anchor="w",
                stretch=column in {"parts", "path"},
            )
        self.tree.tag_configure("done", foreground="#1F8A4C")
        self.tree.tag_configure("failed", foreground="#D93025")
        self.tree.tag_configure("active", foreground="#007AFF")
        self.tree.tag_configure("paused", foreground="#B26A00")

        tree_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        tree_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(
            yscrollcommand=tree_y.set,
            xscrollcommand=tree_x.set,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        details_pad = ttk.Frame(details_card, style="Card.TFrame", padding=(14, 10))
        details_pad.pack(fill="both", expand=True)
        details_pad.grid_columnconfigure(1, weight=1)

        ttk.Label(
            details_pad,
            textvariable=self.detail_title_var,
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        ttk.Label(details_pad, text="角色识别", style="FieldLabel.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=2,
        )
        name_inline = ttk.Frame(details_pad, style="Card.TFrame")
        name_inline.grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(name_inline, textvariable=self.detail_vars["name"]).pack(side="left")
        ttk.Label(name_inline, text=" / ", style="Hint.TLabel").pack(side="left")
        ttk.Label(
            name_inline,
            textvariable=self.detail_vars["romanized"],
            style="Hint.TLabel",
        ).pack(side="left")

        ttk.Label(details_pad, text="ID 信息", style="FieldLabel.TLabel").grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=2,
        )
        id_inline = ttk.Frame(details_pad, style="Card.TFrame")
        id_inline.grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(id_inline, text="角色:", style="Hint.TLabel").pack(side="left")
        ttk.Label(id_inline, textvariable=self.detail_vars["character"]).pack(
            side="left", padx=(3, 12)
        )
        ttk.Label(id_inline, text="头部:", style="Hint.TLabel").pack(side="left")
        ttk.Label(id_inline, textvariable=self.detail_vars["head"]).pack(
            side="left", padx=(3, 0)
        )

        for row, (label, key) in enumerate(
            (("部件列表", "parts"), ("输出位置", "output")),
            start=3,
        ):
            ttk.Label(details_pad, text=label, style="FieldLabel.TLabel").grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 10),
                pady=2,
            )
            value = ttk.Label(
                details_pad,
                textvariable=self.detail_vars[key],
                wraplength=280,
                justify="left",
            )
            value.grid(row=row, column=1, sticky="nw", pady=2)

        meta_start = 5
        tk.Frame(details_pad, bg="#E5E5EA", height=1).grid(
            row=meta_start,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 5),
        )
        ttk.Label(details_pad, text="手工角色名", style="FieldLabel.TLabel").grid(
            row=meta_start + 1,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=2,
        )
        name_row = ttk.Frame(details_pad, style="Card.TFrame")
        name_row.grid(
            row=meta_start + 1,
            column=1,
            sticky="ew",
            pady=2,
        )
        name_row.grid_columnconfigure(0, weight=1)
        self.name_entry = ttk.Entry(
            name_row,
            textvariable=self.name_override_var,
            style="Apple.TEntry",
        )
        self.name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.clear_name_button = ttk.Button(
            name_row,
            text="清空",
            style="Soft.TButton",
            command=self._clear_override_name,
        )
        self.clear_name_button.grid(row=0, column=1, sticky="e")

        ttk.Label(
            details_pad,
            textvariable=self.comment_label_var,
            style="FieldLabel.TLabel",
        ).grid(
            row=meta_start + 2,
            column=0,
            sticky="nw",
            padx=(0, 10),
            pady=(5, 2),
        )
        comment_shell = tk.Frame(details_pad, bg="#D1D1D6", padx=1, pady=1)
        comment_shell.grid(
            row=meta_start + 2,
            column=1,
            sticky="nsew",
            pady=(5, 2),
        )
        self.comment_text = ScrolledText(
            comment_shell,
            height=3,
            width=24,
            wrap="word",
            undo=True,
            relief="flat",
            borderwidth=0,
            bg="#F9F9FB",
            fg="#1D1D1F",
            insertbackground="#1D1D1F",
            font=("Microsoft YaHei UI", 9),
        )
        self.comment_text.pack(fill="both", expand=True)

        ttk.Label(
            details_pad,
            text="支持 {name} {romanized} {id}，留空则不写入",
            style="Hint.TLabel",
            wraplength=280,
            justify="left",
        ).grid(row=meta_start + 3, column=1, sticky="w")

        detail_actions = ttk.Frame(details_pad, style="Card.TFrame")
        detail_actions.grid(
            row=meta_start + 4,
            column=0,
            columnspan=2,
            sticky="e",
            pady=(5, 0),
        )
        self.apply_selected_comment_button = ttk.Button(
            detail_actions,
            text="应用到选中",
            style="Soft.TButton",
            command=self._apply_comment_to_selected,
        )
        self.apply_selected_comment_button.pack(side="left", padx=(0, 6))
        self.apply_all_comment_button = ttk.Button(
            detail_actions,
            text="应用到全部",
            style="Soft.TButton",
            command=self._apply_comment_to_all,
        )
        self.apply_all_comment_button.pack(side="left", padx=(0, 6))
        self.save_default_comment_button = ttk.Button(
            detail_actions,
            text="设为默认",
            style="Soft.TButton",
            command=self._save_default_comment,
        )
        self.save_default_comment_button.pack(side="left")
        details_pad.grid_rowconfigure(meta_start + 2, weight=1, minsize=56)

        bottom_pad = ttk.Frame(bottom_card, style="Card.TFrame", padding=(14, 10))
        bottom_pad.pack(fill="both", expand=True)

        progress_row = ttk.Frame(bottom_pad, style="Card.TFrame")
        progress_row.pack(fill="x")
        self.progress_bar = ttk.Progressbar(
            progress_row,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self.progress_bar.pack(side="left", fill="x", expand=True)
        ttk.Label(
            progress_row,
            textvariable=self.progress_text_var,
            width=6,
            anchor="e",
            font=("Cascadia Mono", 9, "bold"),
        ).pack(side="left", padx=(10, 0))
        ttk.Label(
            progress_row,
            textvariable=self.status_var,
            width=36,
            anchor="w",
            style="FieldLabel.TLabel",
        ).pack(side="left", padx=(12, 8))
        self.toggle_log_button = ttk.Button(
            progress_row,
            text="收起日志",
            style="Soft.TButton",
            command=self._toggle_log,
        )
        self.toggle_log_button.pack(side="right")

        self.log_shell = tk.Frame(bottom_pad, bg="#E5E5EA", padx=1, pady=1)
        self.log_shell.pack(fill="both", expand=True, pady=(9, 0))
        self.log_text = ScrolledText(
            self.log_shell,
            height=3,
            wrap="word",
            state="disabled",
            font=("Cascadia Mono", 9),
            background="#F9F9FB",
            foreground="#2C2C2E",
            relief="flat",
            borderwidth=0,
        )
        self.log_text.pack(fill="both", expand=True)

    def _toggle_log(self) -> None:
        self._log_visible = not self._log_visible
        if self._log_visible:
            self.log_shell.pack(fill="both", expand=True, pady=(9, 0))
            self.toggle_log_button.configure(text="收起日志")
        else:
            self.log_shell.pack_forget()
            self.toggle_log_button.configure(text="展开日志")

    def _refresh_env_badges(self) -> None:
        for child in self.env_badge_frame.winfo_children():
            child.destroy()
        blender_ok = False
        blender_label = f"Blender (>={MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]})"
        try:
            blender_path = _absolute_path(self.blender_var.get(), "Blender")
            probe = probe_blender_executable(blender_path)
            version = probe.get("version")
            if probe.get("supported") and isinstance(version, tuple):
                blender_ok = True
                blender_label = f"Blender {version[0]}.{version[1]}"
            elif isinstance(version, tuple):
                blender_label = f"Blender {version[0]}.{version[1]} (<{MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]})"
        except Exception:
            blender_ok = False

        tools_root = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "Tools"
        ufbx_ok = (tools_root / "ufbx_bridge.dll").is_file() and (tools_root / "ufbx_importer.py").is_file()
        pmx_io_ok = (tools_root / "pmx_io.py").is_file()

        for label, ok in (
            (blender_label, blender_ok),
            ("ufbx FBX", ufbx_ok),
            ("PMX IO", pmx_io_ok),
        ):
            pill = tk.Label(
                self.env_badge_frame,
                text=f"{'●' if ok else '○'} {label}",
                bg="#EAF7EE" if ok else "#FFF1F0",
                fg="#1F8A4C" if ok else "#D93025",
                font=("Microsoft YaHei UI", 8, "bold"),
                padx=9,
                pady=4,
            )
            pill.pack(side="left", padx=(0, 6))

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label, style="FieldLabel.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=4,
        )
        entry = ttk.Entry(parent, textvariable=variable, style="Apple.TEntry")
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(parent, text="选择…", style="Soft.TButton", command=command)
        button.grid(row=row, column=2, padx=(10, 0), pady=4)
        self.path_buttons.append(button)
        return entry

    def _bind_events(self) -> None:
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selection)
        self.tree.bind("<Double-1>", self._on_tree_double_click)
        self.name_override_var.trace_add("write", self._on_name_changed)
        self.blender_var.trace_add("write", lambda *_: self._refresh_env_badges())
        self.comment_text.bind("<<Modified>>", self._on_comment_changed)

    def _on_body_mapped(self, _event: tk.Event[Any] | None = None) -> None:
        if self._sash_initialized:
            return
        width = self._body_paned.winfo_width()
        if width <= 100:
            return
        self._sash_initialized = True
        try:
            self._body_paned.sash_place(0, int(width * 5 / 8), 0)
        except Exception:
            pass

    def _on_root_configure(self, event: tk.Event[Any]) -> None:
        if event.widget is self.root:
            self._last_configure_at = time.monotonic()

    def _browse_blender(self) -> None:
        initial = self._initial_directory(self.blender_var.get())
        selected = filedialog.askopenfilename(
            parent=self.root,
            title=f"选择 Blender 可执行文件 (>={MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]})",
            initialdir=initial,
            filetypes=(("Blender", "blender.exe"), ("可执行文件", "*.exe"),),
        )
        if selected:
            self.blender_var.set(str(Path(selected).resolve(strict=False)))

    def _browse_downloader(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="选择下载器任务文件夹",
            initialdir=self._initial_directory(self.downloader_var.get()),
            mustexist=True,
        )
        if selected:
            self.downloader_var.set(str(Path(selected).resolve(strict=False)))

    def _browse_output_root(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="选择输出根目录",
            initialdir=self._initial_directory(self.output_root_var.get()),
            mustexist=False,
        )
        if selected:
            self.output_root_var.set(str(Path(selected).resolve(strict=False)))

    def _initial_directory(self, raw: str) -> str:
        try:
            path = _absolute_path(raw, "目录")
        except ValueError:
            return str(APP_DIR)
        if path.is_file():
            path = path.parent
        return str(path if path.exists() else APP_DIR)

    def _choose_tasks(self, multiple: bool) -> None:
        if self._building:
            return
        try:
            task_root = _absolute_path(
                self.downloader_var.get(),
                "下载器任务文件夹",
            )
            initial = task_root if task_root.is_dir() else APP_DIR
        except ValueError:
            initial = APP_DIR

        total_added = 0
        if multiple:
            picked = _pick_multiple_directories_win32(
                self.root.winfo_id(),
                "批量选择任务文件夹（可按 Ctrl / Shift 多选，或直接选下载根目录）",
                initial,
            )
            if picked is None:
                selected = filedialog.askdirectory(
                    parent=self.root,
                    title="选择任务文件夹或包含多个任务的根目录",
                    initialdir=str(initial),
                    mustexist=True,
                )
                picked = [selected] if selected else []
            if picked:
                before = len(self._tasks)
                self._add_task_directories(_expand_task_directories(picked))
                total_added = len(self._tasks) - before
        else:
            selected = filedialog.askdirectory(
                parent=self.root,
                title="选择任务文件夹",
                initialdir=str(initial),
                mustexist=True,
            )
            if selected:
                before = len(self._tasks)
                self._add_task_directories([selected])
                total_added = len(self._tasks) - before
        if total_added:
            self.status_var.set(f"新增 {total_added} 个任务，正在识别")

    def _add_task_directories(self, directories: list[str]) -> None:
        added_keys: list[str] = []
        for raw in directories:
            try:
                task_dir = _absolute_path(raw, "任务文件夹")
            except ValueError as exc:
                messagebox.showerror("任务文件夹", str(exc), parent=self.root)
                continue
            if not task_dir.is_dir():
                messagebox.showerror(
                    "任务文件夹",
                    f"文件夹不存在：{task_dir}",
                    parent=self.root,
                )
                continue
            key = _path_key(task_dir)
            duplicate = next(
                (
                    task
                    for task in self._tasks.values()
                    if _path_key(task.task_dir) == key
                ),
                None,
            )
            if duplicate is not None:
                self.tree.selection_set(duplicate.key)
                self.tree.focus(duplicate.key)
                self._append_log(f"任务已在队列中：{task_dir}")
                continue
            task_key = uuid.uuid4().hex
            task = StudioTask(
                key=task_key,
                task_dir=task_dir,
                comment=self._default_comment,
            )
            self._tasks[task_key] = task
            self.tree.insert("", "end", iid=task_key, values=self._tree_values(task))
            added_keys.append(task_key)
            self._inspect_jobs.put((task_key, task_dir))
            self._append_log(f"添加任务：{task_dir}")

        if added_keys:
            first = added_keys[0]
            self.tree.selection_set(first)
            self.tree.focus(first)
            self.tree.see(first)
            self._refresh_actions()

    def _remove_selected(self) -> None:
        if self._building:
            return
        selected = list(self.tree.selection())
        if not selected:
            return
        paths = [self._tasks[key].task_dir for key in selected if key in self._tasks]
        if not messagebox.askyesno(
            "移除任务",
            f"从队列移除 {len(selected)} 个任务？\n源任务文件夹不会删除。",
            parent=self.root,
        ):
            return
        for key in selected:
            self._tasks.pop(key, None)
            if self.tree.exists(key):
                self.tree.delete(key)
        self._selected_key = ""
        self._show_details_for_selection()
        for path in paths:
            self._append_log(f"已从队列移除：{path}")
        self._refresh_actions()

    def _build_single(self) -> None:
        selected = list(self.tree.selection())
        if len(selected) != 1:
            messagebox.showinfo(
                "单个制作",
                "请选择一个任务后再开始。",
                parent=self.root,
            )
            return
        task = self._tasks.get(selected[0])
        if task is None:
            return
        self._start_build([task], "单个制作")

    def _build_batch(self) -> None:
        jobs = [
            task
            for task in self._tasks.values()
            if task.status != "已完成"
        ]
        if not jobs:
            messagebox.showinfo(
                "开始批量",
                "队列中没有未完成任务。",
                parent=self.root,
            )
            return
        self._start_build(jobs, "批量制作")

    def _build_context(self) -> tuple[Path, Path]:
        blender = _absolute_path(self.blender_var.get(), "Blender 路径")
        blender = validate_blender_executable(blender)
        output_root = _absolute_path(
            self.output_root_var.get(),
            "输出根目录",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        if not output_root.is_dir():
            raise ValueError(f"输出根目录不可用：{output_root}")
        self.blender_var.set(str(blender))
        self.output_root_var.set(str(output_root))
        return blender, output_root

    def _start_build(self, tasks: list[StudioTask], label: str) -> None:
        if self._building:
            return
        self._flush_comment_editor()
        try:
            blender, output_root = self._build_context()
        except (OSError, ValueError) as exc:
            messagebox.showerror("路径检查", str(exc), parent=self.root)
            return

        jobs: list[BuildJob] = []
        for task in tasks:
            task_dir = task.task_dir.resolve(strict=False)
            if not task_dir.is_dir():
                messagebox.showerror(
                    "任务文件夹",
                    f"文件夹不存在：{task_dir}",
                    parent=self.root,
                )
                return
            jobs.append(
                BuildJob(
                    task_key=task.key,
                    task_dir=task_dir,
                    name_override=task.override_name.strip(),
                    comment=task.comment.rstrip(),
                )
            )
        if not jobs:
            return

        self._building = True
        self._stop_event = threading.Event()
        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        self.status_var.set(f"{label}：准备 {len(jobs)} 个任务")
        self._set_path_editing(False)
        self._set_detail_editing(False)
        self._refresh_actions()
        self._append_log(
            f"开始{label}：{len(jobs)} 个任务；输出根目录 {output_root}"
        )
        worker = threading.Thread(
            target=self._build_worker,
            args=(jobs, output_root, blender, self._stop_event),
            name="scsp-build",
            daemon=True,
        )
        worker.start()

    def _build_worker(
        self,
        jobs: list[BuildJob],
        output_root: Path,
        blender: Path,
        stop_event: threading.Event,
    ) -> None:
        total = len(jobs)
        succeeded = 0
        failed = 0
        stopped = False
        pipeline, load_error = self._load_pipeline()

        if pipeline is None:
            for job in jobs:
                self._events.put(
                    ("task_failed", job.task_key, load_error or "流水线不可用")
                )
            self._events.put(
                (
                    "batch_finished",
                    {
                        "total": total,
                        "succeeded": 0,
                        "failed": total,
                        "stopped": False,
                    },
                )
            )
            return

        for index, job in enumerate(jobs):
            if stop_event.is_set():
                stopped = True
                for remaining in jobs[index:]:
                    self._events.put(
                        (
                            "task_skipped",
                            remaining.task_key,
                            "当前任务完成后已停止",
                        )
                    )
                break

            self._events.put(("task_building", job.task_key))
            self._events.put(
                (
                    "overall_progress",
                    index / total * 100,
                    f"正在制作 {index + 1}/{total}：{job.task_dir.name}",
                )
            )
            last_emit = [0.0, -1]

            def progress_callback(*args: Any, **kwargs: Any) -> None:
                percent, message = self._parse_progress(args, kwargs)
                now = time.monotonic()
                should_emit = (
                    message
                    or percent in {0, 100}
                    or now - last_emit[0] >= 0.15
                )
                if not should_emit:
                    return
                last_emit[0] = now
                overall = (index + percent / 100.0) / total * 100.0
                self._events.put(
                    ("task_progress", job.task_key, percent, message)
                )
                self._events.put(
                    ("overall_progress", overall, message)
                )

            try:
                result = pipeline.build_task(
                    str(job.task_dir),
                    str(output_root),
                    str(blender),
                    name_override=job.name_override or None,
                    comment=job.comment,
                    progress=progress_callback,
                )
                if not isinstance(result, dict):
                    raise TypeError(
                        "build_task 必须返回包含 pmx、output_dir 的字典"
                    )
                pmx_value = result.get("pmx")
                output_value = result.get("output_dir")
                if output_value:
                    output_path = Path(output_value)
                    if not output_path.is_absolute():
                        output_path = output_root / output_path
                    output_dir = output_path.resolve(strict=False)
                else:
                    output_dir = None
                if pmx_value:
                    pmx_path = Path(pmx_value)
                    if not pmx_path.is_absolute():
                        pmx_path = (output_dir or output_root) / pmx_path
                    pmx = pmx_path.resolve(strict=False)
                else:
                    pmx = None
                if output_dir is None and pmx is not None:
                    output_dir = pmx.parent
                self._events.put(
                    ("task_result", job.task_key, pmx, output_dir)
                )
                succeeded += 1
            except Exception as exc:
                message = _friendly_error(exc)
                self._events.put(("task_failed", job.task_key, message))
                self._events.put(
                    (
                        "log",
                        f"任务失败：{job.task_dir}\n{traceback.format_exc()}",
                    )
                )
                failed += 1
            self._events.put(
                (
                    "overall_progress",
                    (index + 1) / total * 100,
                    f"已完成 {index + 1}/{total}",
                )
            )

        self._events.put(
            (
                "batch_finished",
                {
                    "total": total,
                    "succeeded": succeeded,
                    "failed": failed,
                    "stopped": stopped,
                },
            )
        )

    def _parse_progress(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[float, str]:
        numbers: list[float] = []
        messages: list[str] = []
        current: float | None = None
        total: float | None = None
        direct_percent: float | None = None

        def read_number(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                text = value.strip().replace("%", "")
                try:
                    return float(text)
                except ValueError:
                    return None
            return None

        def absorb(payload: dict[str, Any]) -> None:
            nonlocal current, total, direct_percent
            for key in ("percent", "progress", "value", "fraction"):
                if key in payload:
                    parsed = read_number(payload[key])
                    if parsed is not None:
                        direct_percent = parsed
                        break
            if "current" in payload and "total" in payload:
                parsed_current = read_number(payload["current"])
                parsed_total = read_number(payload["total"])
                if parsed_current is not None and parsed_total is not None:
                    current, total = parsed_current, parsed_total
            for key in ("message", "stage", "detail", "text", "status"):
                if payload.get(key):
                    messages.append(str(payload[key]))

        for arg in args:
            if isinstance(arg, dict):
                absorb(arg)
                continue
            parsed = read_number(arg)
            if parsed is not None:
                numbers.append(parsed)
            elif arg is not None and str(arg).strip():
                messages.append(str(arg))
        absorb(kwargs)

        if current is None and len(numbers) >= 2:
            current, total = numbers[0], numbers[1]
            if len(numbers) >= 3 and direct_percent is None:
                direct_percent = numbers[2]
        elif direct_percent is None and numbers:
            direct_percent = numbers[0]

        if current is not None and total is not None:
            percent = 0.0 if total <= 0 else current / total * 100.0
        elif direct_percent is not None:
            percent = direct_percent * 100.0 if 0 < direct_percent <= 1 else direct_percent
        else:
            percent = 0.0
        percent = max(0.0, min(100.0, percent))
        message = " · ".join(
            _single_line(item, "") for item in messages if _single_line(item, "")
        )
        return percent, message

    def _request_stop_after_current(self) -> None:
        if not self._building or self._stop_event is None:
            return
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self.status_var.set("已请求停止：当前任务完成后停止后续任务")
        self._append_log("已请求停止后续任务；当前任务会继续完成。")
        self._refresh_actions()

    def _open_output_directory(self) -> None:
        selected = list(self.tree.selection())
        target: Path | None = None
        for key in selected:
            task = self._tasks.get(key)
            if task and task.output_dir:
                target = task.output_dir
                break
        if target is None:
            try:
                candidate = _absolute_path(
                    self.output_root_var.get(),
                    "输出根目录",
                )
            except ValueError as exc:
                messagebox.showerror("打开输出目录", str(exc), parent=self.root)
                return
            if candidate.exists():
                target = candidate
        if target is None or not target.exists():
            messagebox.showinfo(
                "打开输出目录",
                "所选任务还没有输出目录，输出根目录也尚未创建。",
                parent=self.root,
            )
            return
        try:
            if os.name == "nt":
                os.startfile(str(target))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as exc:
            messagebox.showerror(
                "打开输出目录",
                f"无法打开 {target}\n{_friendly_error(exc)}",
                parent=self.root,
            )

    def _on_tree_selection(self, _event: tk.Event[Any] | None = None) -> None:
        self._flush_comment_editor()
        selection = list(self.tree.selection())
        self._selected_key = selection[0] if len(selection) == 1 else ""
        self._show_details_for_selection()
        self._refresh_actions()

    def _on_tree_double_click(self, event: tk.Event[Any]) -> None:
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.tree.focus(row)
            self._open_output_directory()

    def _show_details_for_selection(self) -> None:
        selection = list(self.tree.selection())
        task = self._tasks.get(self._selected_key) if len(selection) == 1 else None
        self._syncing_details = True
        try:
            if task is None:
                if len(selection) > 1:
                    self.detail_title_var.set(f"已选择 {len(selection)} 个任务")
                    self.comment_label_var.set("批量 PMX 说明")
                else:
                    self.detail_title_var.set("尚未选择任务")
                    self.comment_label_var.set("默认 PMX 说明")
                for variable in self.detail_vars.values():
                    variable.set("—")
                self.name_override_var.set("")
                self._set_comment(self._default_comment)
            else:
                self.detail_title_var.set(task.display_name)
                self.detail_vars["name"].set(task.name or "—")
                self.detail_vars["romanized"].set(task.romanized_name or "—")
                self.detail_vars["character"].set(task.character_id or "—")
                self.detail_vars["head"].set(task.head_id or "—")
                self.detail_vars["parts"].set(task.parts_display)
                self.detail_vars["output"].set(
                    str(task.output_dir) if task.output_dir else "—"
                )
                self.name_override_var.set(task.override_name)
                self.comment_label_var.set("PMX 说明")
                self._set_comment(task.comment)
        finally:
            self._syncing_details = False
        detail_enabled = task is not None and not self._building
        self._set_detail_editing(detail_enabled)

    def _set_comment(self, value: str) -> None:
        self.comment_text.configure(state="normal")
        self.comment_text.delete("1.0", "end")
        self.comment_text.insert("1.0", value)
        self.comment_text.edit_modified(False)
        self.comment_text.configure(
            state="disabled" if self._building else "normal"
        )

    def _on_name_changed(self, *_args: object) -> None:
        if self._syncing_details or self._building or not self._selected_key:
            return
        task = self._tasks.get(self._selected_key)
        if task is None:
            return
        task.override_name = self.name_override_var.get()
        self.detail_title_var.set(task.display_name)
        self._update_task_row(task)

    def _read_comment_editor(self) -> str:
        """Read current PMX comment text and clear the widget modified flag."""
        if not hasattr(self, "comment_text"):
            return self._default_comment
        text = self.comment_text.get("1.0", "end-1c")
        self.comment_text.edit_modified(False)
        return text

    def _flush_comment_editor(self) -> None:
        """Sync text widget content into either the active task or default comment."""
        if self._syncing_details or self._building or not hasattr(self, "comment_text"):
            return
        text = self._read_comment_editor()
        task = self._tasks.get(self._selected_key) if self._selected_key else None
        if task is not None:
            task.comment = text
        else:
            self._default_comment = text

    def _on_comment_changed(self, _event: tk.Event[Any] | None = None) -> None:
        if not self.comment_text.edit_modified():
            return
        self._flush_comment_editor()

    def _persist_settings(self, path: Path | None = None) -> None:
        self._settings["default_pmx_comment"] = self._default_comment
        save_studio_settings(self._settings, path=path)

    def _save_default_comment(self) -> None:
        if self._building:
            return
        self._default_comment = self._read_comment_editor()
        try:
            self._persist_settings()
        except OSError as exc:
            messagebox.showerror("保存默认说明", str(exc), parent=self.root)
            return
        self._append_log("已将当前内容设为默认 PMX 说明（新建任务将自动带入）。")

    def _apply_comment_to_selected(self) -> None:
        if self._building:
            return
        selection = [key for key in self.tree.selection() if key in self._tasks]
        if not selection:
            return
        text = self._read_comment_editor()
        for key in selection:
            self._tasks[key].comment = text
        self._append_log(f"已将当前 PMX 说明应用到选中的 {len(selection)} 个任务。")

    def _apply_comment_to_all(self) -> None:
        if self._building:
            return
        text = self._read_comment_editor()
        for task in self._tasks.values():
            task.comment = text
        self._append_log(f"已将当前 PMX 说明应用到全部 {len(self._tasks)} 个任务。")

    def _clear_override_name(self) -> None:
        if self._building or not self._selected_key:
            return
        self.name_override_var.set("")

    def _refresh_actions(self) -> None:
        selection = list(self.tree.selection())
        unfinished = any(
            task.status != "已完成" for task in self._tasks.values()
        )
        if hasattr(self, "queue_count_var"):
            self.queue_count_var.set(f"{len(self._tasks)} 项")
        self._set_enabled(self.add_button, not self._building)
        self._set_enabled(self.add_many_button, not self._building)
        self._set_enabled(
            self.remove_button,
            bool(selection) and not self._building,
        )
        self._set_enabled(
            self.single_button,
            len(selection) == 1 and not self._building,
        )
        self._set_enabled(
            self.batch_button,
            unfinished and not self._building,
        )
        stop_pressed = bool(self._stop_event and self._stop_event.is_set())
        self._set_enabled(
            self.stop_button,
            self._building and not stop_pressed,
        )
        self._set_enabled(self.open_button, True)
        self._set_enabled(
            self.apply_selected_comment_button,
            bool(selection) and not self._building,
        )
        self._set_enabled(
            self.apply_all_comment_button,
            bool(self._tasks) and not self._building,
        )
        self._set_enabled(
            self.save_default_comment_button,
            not self._building,
        )

    def _set_path_editing(self, enabled: bool) -> None:
        for entry in self.path_entries:
            entry.configure(state="normal" if enabled else "disabled")
        for button in self.path_buttons:
            self._set_enabled(button, enabled)

    def _set_detail_editing(self, enabled: bool) -> None:
        selection = list(self.tree.selection()) if hasattr(self, "tree") else []
        self.name_entry.configure(state="normal" if enabled else "disabled")
        self.comment_text.configure(
            state="disabled" if self._building else "normal"
        )
        self._set_enabled(self.clear_name_button, enabled)
        self._set_enabled(
            self.apply_selected_comment_button,
            bool(selection) and not self._building,
        )
        self._set_enabled(
            self.apply_all_comment_button,
            bool(self._tasks) and not self._building,
        )
        self._set_enabled(
            self.save_default_comment_button,
            not self._building,
        )

    @staticmethod
    def _set_enabled(widget: ttk.Widget, enabled: bool) -> None:
        widget.configure(state="normal" if enabled else "disabled")

    def _tree_values(self, task: StudioTask) -> tuple[str, ...]:
        return (
            task.status,
            task.display_name,
            task.romanized_name or "—",
            task.character_id or "—",
            task.head_id or "—",
            task.parts_display,
            f"{task.progress}%",
            str(task.task_dir),
        )

    def _update_task_row(self, task: StudioTask) -> None:
        if not self.tree.exists(task.key):
            return
        if task.status == "已完成":
            tag = "done"
        elif task.status in {"失败", "识别失败"}:
            tag = "failed"
        elif task.status == "制作中":
            tag = "active"
        elif "停止" in task.status:
            tag = "paused"
        else:
            tag = ""
        self.tree.item(
            task.key,
            values=self._tree_values(task),
            tags=(tag,) if tag else (),
        )

    def _load_pipeline(self) -> tuple[Any | None, str | None]:
        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline, None
            try:
                importlib.invalidate_caches()
                module = importlib.import_module("studio_pipeline")
                missing = [
                    name
                    for name in ("inspect_task", "build_task")
                    if not callable(getattr(module, name, None))
                ]
                if missing:
                    return None, (
                        "studio_pipeline 缺少接口：" + ", ".join(missing)
                    )
                self._pipeline = module
                return self._pipeline, None
            except Exception as exc:
                return None, (
                    "无法导入同目录的 studio_pipeline.py："
                    f"{_friendly_error(exc)}"
                )

    def _inspector_loop(self) -> None:
        while not self._closing:
            try:
                item = self._inspect_jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return
            task_key, task_dir = item
            pipeline, error = self._load_pipeline()
            if pipeline is None:
                self._events.put(("inspect_error", task_key, error))
                continue
            try:
                info = pipeline.inspect_task(str(task_dir))
                if not isinstance(info, dict):
                    raise TypeError("inspect_task 必须返回字典")
                self._events.put(("inspect_ok", task_key, info))
            except Exception as exc:
                self._events.put(
                    ("inspect_error", task_key, _friendly_error(exc))
                )

    def _poll_events(self) -> None:
        if self._closing:
            return
        # Defer heavy UI mutations briefly while the user is actively moving/resizing the window.
        if time.monotonic() - getattr(self, "_last_configure_at", 0.0) < 0.14:
            self.root.after(140, self._poll_events)
            return
        for _ in range(300):
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        inspecting = (
            not self._inspect_jobs.empty()
            or any(task.status == "识别中" for task in self._tasks.values())
        )
        interval = EVENT_POLL_ACTIVE_MS if (self._building or inspecting) else EVENT_POLL_IDLE_MS
        self.root.after(interval, self._poll_events)

    def _handle_event(self, event: tuple[Any, ...]) -> None:
        event_type = event[0]
        if event_type == "inspect_ok":
            _, task_key, info = event
            task = self._tasks.get(task_key)
            if task is None:
                return
            task.name = _single_line(info.get("name"), "")
            task.romanized_name = _single_line(
                info.get("romanized_name"),
                "",
            )
            task.character_id = _single_line(
                info.get("character_id"),
                "",
            )
            task.head_id = _single_line(info.get("head_id"), "")
            task.parts = _parts_from_value(info.get("parts"))
            if task.status == "识别中":
                task.status = "待制作"
            self._update_task_row(task)
            if task.key == self._selected_key:
                self._show_details_for_selection()
            self._append_log(
                f"识别完成：{task.display_name} | 角色 ID "
                f"{task.character_id or '未知'} | 头部 ID "
                f"{task.head_id or '未知'}"
            )
            self._refresh_actions()
        elif event_type == "inspect_error":
            _, task_key, error = event
            task = self._tasks.get(task_key)
            if task is None:
                return
            if task.status == "识别中":
                task.status = "识别失败"
            task.error = str(error)
            self._update_task_row(task)
            self._append_log(f"识别失败：{task.task_dir} | {error}")
            self._refresh_actions()
        elif event_type == "task_building":
            _, task_key = event
            task = self._tasks.get(task_key)
            if task is not None:
                task.status = "制作中"
                task.progress = 0
                task.error = ""
                self._update_task_row(task)
        elif event_type == "task_progress":
            _, task_key, percent, message = event
            task = self._tasks.get(task_key)
            if task is not None:
                task.progress = int(round(float(percent)))
                task.detail = str(message or "")
                self._update_task_row(task)
            if message:
                self._append_log(f"{task.task_dir.name if task else task_key}：{message}")
        elif event_type == "task_result":
            _, task_key, pmx, output_dir = event
            task = self._tasks.get(task_key)
            if task is not None:
                task.status = "已完成"
                task.progress = 100
                task.pmx = pmx
                task.output_dir = output_dir
                task.error = ""
                self._update_task_row(task)
                if task.key == self._selected_key:
                    self._show_details_for_selection()
                self._append_log(
                    f"任务完成：{task.display_name} | "
                    f"{pmx if pmx else output_dir}"
                )
            self._refresh_actions()
        elif event_type == "task_failed":
            _, task_key, error = event
            task = self._tasks.get(task_key)
            if task is not None:
                task.status = "失败"
                task.error = str(error)
                self._update_task_row(task)
                self._append_log(f"任务失败：{task.task_dir} | {error}")
            self._refresh_actions()
        elif event_type == "task_skipped":
            _, task_key, reason = event
            task = self._tasks.get(task_key)
            if task is not None:
                task.status = "已停止"
                task.detail = str(reason)
                self._update_task_row(task)
                self._append_log(f"未开始：{task.task_dir} | {reason}")
        elif event_type == "overall_progress":
            _, value, message = event
            numeric = max(0.0, min(100.0, float(value)))
            self.progress_var.set(numeric)
            self.progress_text_var.set(f"{numeric:.0f}%")
            self.status_var.set(str(message or "制作中"))
        elif event_type == "log":
            _, message = event
            self._append_log(str(message).rstrip())
        elif event_type == "batch_finished":
            _, summary = event
            self._finish_build(summary)

    def _finish_build(self, summary: dict[str, Any]) -> None:
        self._building = False
        self._stop_event = None
        self._set_path_editing(True)
        self._refresh_actions()
        self._show_details_for_selection()
        total = int(summary.get("total", 0))
        succeeded = int(summary.get("succeeded", 0))
        failed = int(summary.get("failed", 0))
        stopped = bool(summary.get("stopped"))
        skipped = total - succeeded - failed
        if stopped or skipped > 0:
            status = (
                f"已停止：完成 {succeeded}，失败 {failed}，未开始 {max(0, skipped)}"
            )
        else:
            status = f"本批结束：完成 {succeeded}，失败 {failed}"
        self.status_var.set(status)
        self._append_log(status)
        if failed:
            self.progress_text_var.set(
                f"{self.progress_var.get():.0f}%"
            )
        elif not stopped:
            self.progress_var.set(100)
            self.progress_text_var.set("100%")

    def _append_log(self, message: str) -> None:
        clean = message.replace("\x00", "").rstrip()
        if not clean:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {clean}\n")
        self._log_lines += clean.count("\n") + 1
        if self._log_lines > 3000:
            self.log_text.delete("1.0", "800.0")
            self._log_lines = max(0, self._log_lines - 799)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self._building and not messagebox.askyesno(
            "退出工具",
            "后台仍有任务在制作。现在退出会中断当前进程，"
            "可能留下不完整输出。\n仍要退出吗？",
            parent=self.root,
        ):
            return
        self._flush_comment_editor()
        try:
            self._persist_settings()
        except OSError:
            pass
        self._closing = True
        try:
            self._inspect_jobs.put_nowait(None)
        except queue.Full:
            pass
        self.root.destroy()


def main() -> None:
    enable_windows_dpi_awareness()
    root = tk.Tk()
    StudioGui(root)
    root.mainloop()


def headless_build() -> int:
    """Run the packaged backend without opening a Tk window."""
    import argparse
    import json
    from studio_pipeline import build_task
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless-build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--report", required=True)
    parser.add_argument("--name-override", default=None)
    parser.add_argument("--comment", default="")
    args = parser.parse_args()
    code = 0
    try:
        result = build_task(
            args.headless_build,
            args.output,
            args.blender,
            name_override=args.name_override,
            comment=args.comment,
        )
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        code = 1
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    if "--headless-build" in sys.argv:
        raise SystemExit(headless_build())
    main()
