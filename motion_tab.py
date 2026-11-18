"""Motion -> VMD conversion tab for StudioGui."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

inspect_task: Any = None
build_task: Any = None


def _resource_root() -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    if not (root / "Resources").is_dir():
        root = Path(sys.prefix)
    return root


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_motion_pipeline() -> tuple[Any, Any]:
    """Resolve inspect_task/build_task lazily so missing motion deps never break PMX tab."""
    if callable(inspect_task) and callable(build_task):
        return inspect_task, build_task
    try:
        import motion_pipeline as mp
    except ModuleNotFoundError as exc:
        missing = exc.name or "numpy / UnityPy"
        raise RuntimeError(
            f"缺少动作转换依赖: {missing}。请先运行 pip install -r requirements.txt。"
        ) from exc
    return (
        inspect_task if callable(inspect_task) else mp.inspect_task,
        build_task if callable(build_task) else mp.build_task,
    )


def _create_card(parent: tk.Misc) -> ttk.Frame:
    outer = tk.Frame(parent, bg="#E5E5EA", padx=1, pady=1)
    inner = ttk.Frame(outer, style="Card.TFrame")
    inner.pack(fill="both", expand=True)
    inner._card_outer = outer  # type: ignore[attr-defined]
    return inner


class MotionTab(ttk.Frame):
    """Embedded tab providing downloader task -> MMD VMD conversion."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        settings: dict[str, Any] | None = None,
        save_settings: Callable[[], None] | None = None,
        default_output_root: Path | None = None,
    ) -> None:
        super().__init__(master, style="TFrame")
        self.bus: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.running = False
        self.last_output: str | None = None
        self.info: dict[str, Any] | None = None
        self.targets: dict[int, str] = {}
        self._closing = False
        self._poll_after_id: str | None = None

        app_dir = (
            Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
        fallback_output = str(default_output_root or (app_dir / "Outputs"))

        self._external_settings = settings
        self._save_callback = save_settings
        self.settings_path = Path(
            os.environ.get("SCSP_MOTION_SETTINGS", str(app_dir / "settings.json"))
        )
        if settings is not None:
            state = settings
            self.task = tk.StringVar(value=str(state.get("motion_task", "") or ""))
            self.output = tk.StringVar(
                value=str(state.get("motion_output") or fallback_output)
            )
            self.pmx = tk.StringVar(value=str(state.get("motion_pmx", "") or ""))
            self.rules = tk.StringVar(value=str(state.get("motion_rules", "") or ""))
        else:
            try:
                state = _read_json(self.settings_path) if self.settings_path.is_file() else {}
            except (OSError, ValueError):
                state = {}
            self.task = tk.StringVar(value=str(state.get("task", "") or ""))
            self.output = tk.StringVar(value=str(state.get("output") or fallback_output))
            self.pmx = tk.StringVar(value=str(state.get("pmx", "") or ""))
            self.rules = tk.StringVar(value=str(state.get("rules", "") or ""))

        self.status = tk.StringVar(value="就绪")
        self.controls: list[Any] = []
        self._build_ui()
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._poll_after_id = self.after(120, self.poll)

    def _on_destroy(self, event: tk.Event[Any]) -> None:
        if event.widget is self:
            self.stop_polling()

    def _build_ui(self) -> None:
        paths_card = _create_card(self)
        paths_card._card_outer.pack(fill="x", padx=18, pady=(14, 10))  # type: ignore[attr-defined]
        form = ttk.Frame(paths_card, style="Card.TFrame", padding=(16, 12))
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="工作区与路径配置", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        rows = [
            ("歌曲任务目录", self.task, "dir"),
            ("默认目标 PMX", self.pmx, "pmx"),
            ("输出目录", self.output, "dir"),
            ("自定义规则", self.rules, "json"),
        ]
        for row, (label, var, kind) in enumerate(rows, start=1):
            ttk.Label(form, text=label, style="FieldLabel.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=4
            )
            entry = ttk.Entry(form, textvariable=var, style="Apple.TEntry")
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            button = ttk.Button(
                form,
                text="选择…",
                style="Soft.TButton",
                command=lambda v=var, k=kind: self.browse(v, k),
            )
            button.grid(row=row, column=2, padx=(10, 0), pady=4)
            self.controls.extend((entry, button))

        bottom_card = _create_card(self)
        bottom_card._card_outer.pack(side="bottom", fill="x", padx=18, pady=(0, 14))  # type: ignore[attr-defined]

        table_card = _create_card(self)
        table_card._card_outer.pack(side="top", fill="both", expand=True, padx=18, pady=(0, 10))  # type: ignore[attr-defined]
        table_pad = ttk.Frame(table_card, style="Card.TFrame", padding=(14, 12))
        table_pad.pack(fill="both", expand=True)

        actions = ttk.Frame(table_pad, style="Card.TFrame")
        actions.pack(fill="x", pady=(0, 10))
        ttk.Label(actions, text="站位列表", style="Section.TLabel").pack(
            side="left", padx=(0, 14)
        )
        for label, command in [
            ("检查任务", self.inspect),
            ("设置选中站位 PMX", self.assign),
            ("清除选中站位 PMX", self.clear_assign),
            ("全选站位", self.select_all),
            ("查看默认规则", self.show_rules),
        ]:
            button = ttk.Button(actions, text=label, style="Soft.TButton", command=command)
            button.pack(side="left", padx=(0, 6))
            self.controls.append(button)

        tree_shell = tk.Frame(table_pad, bg="#E5E5EA", padx=1, pady=1)
        tree_shell.pack(fill="both", expand=True)
        tree_frame = ttk.Frame(tree_shell, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("slot", "character", "duration", "pmx"),
            show="headings",
            height=7,
            selectmode="extended",
        )
        for col, title, width in [
            ("slot", "站位", 70),
            ("character", "角色编号", 90),
            ("duration", "时长（秒）", 100),
            ("pmx", "目标 PMX", 630),
        ]:
            self.tree.heading(col, text=title, anchor="w")
            self.tree.column(col, width=width, minwidth=60, anchor="w", stretch=(col == "pmx"))

        tree_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        tree_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        bottom_pad = ttk.Frame(bottom_card, style="Card.TFrame", padding=(14, 10))
        bottom_pad.pack(fill="both", expand=True)

        progress_row = ttk.Frame(bottom_pad, style="Card.TFrame")
        progress_row.pack(fill="x")
        self.bar = ttk.Progressbar(
            progress_row, maximum=100, mode="determinate", style="Horizontal.TProgressbar"
        )
        self.bar.pack(side="left", fill="x", expand=True)
        ttk.Label(
            progress_row,
            textvariable=self.status,
            width=44,
            anchor="w",
            style="FieldLabel.TLabel",
        ).pack(side="left", padx=(12, 10))
        self.start = ttk.Button(
            progress_row, text="开始转换", style="Primary.TButton", command=self.build
        )
        self.start.pack(side="right")
        self.open_btn = ttk.Button(
            progress_row, text="打开输出目录", style="Soft.TButton", command=self.open_output
        )
        self.open_btn.pack(side="right", padx=(0, 8))

        log_shell = tk.Frame(bottom_pad, bg="#E5E5EA", padx=1, pady=1)
        log_shell.pack(fill="both", expand=True, pady=(9, 0))
        self.log = ScrolledText(
            log_shell,
            height=5,
            font=("Cascadia Mono", 9),
            state="disabled",
            wrap="word",
            background="#F9F9FB",
            foreground="#2C2C2E",
            relief="flat",
            borderwidth=0,
        )
        self.log.pack(fill="both", expand=True)

    def read_settings(self) -> dict[str, str]:
        return {
            "motion_task": self.task.get(),
            "motion_output": self.output.get(),
            "motion_pmx": self.pmx.get(),
            "motion_rules": self.rules.get(),
        }

    def _save_state(self) -> None:
        if self._external_settings is not None:
            self._external_settings.update(self.read_settings())
            if self._save_callback is not None:
                self._save_callback()
            return
        _write_json(
            self.settings_path,
            {k: getattr(self, k).get() for k in ("task", "output", "pmx", "rules")},
        )

    def browse(self, var: tk.StringVar, kind: str) -> None:
        if kind == "dir":
            value = filedialog.askdirectory(parent=self)
        else:
            value = filedialog.askopenfilename(
                parent=self,
                filetypes=[("PMX", "*.pmx")] if kind == "pmx" else [("JSON", "*.json")],
            )
        if value:
            var.set(value)

    def append(self, text: object) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", str(text) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def inspect(self) -> None:
        raw_task = self.task.get().strip()
        if not raw_task:
            self.info = None
            messagebox.showerror("缺少任务目录", "请先选择歌曲任务目录。", parent=self)
            return
        try:
            inspect_fn, _ = _load_motion_pipeline()
            self.info = inspect_fn(raw_task)
            self.targets.clear()
            self.tree.delete(*self.tree.get_children())
            for s in self.info["slots"]:
                key = str(s["slot"])
                self.tree.insert(
                    "",
                    "end",
                    iid=key,
                    values=(s["slot"], s["character_id"], f"{s['duration']:.3f}", "默认 / 通用 MMD"),
                )
            self.select_all()
            self.status.set(
                f"{self.info['song']} · {len(self.info['slots'])} 个站位 · 口型 {len(self.info['mouths'])} 份"
            )
            for warning in self.info["warnings"]:
                self.append(warning)
        except Exception as error:
            self.info = None
            messagebox.showerror("任务检查失败", str(error), parent=self)

    def select_all(self) -> None:
        self.tree.selection_set(self.tree.get_children())

    def assign(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("PMX", "*.pmx")])
        if path:
            for key in selected:
                self.targets[int(key)] = path
                values = list(self.tree.item(key, "values"))
                values[-1] = path
                self.tree.item(key, values=values)

    def clear_assign(self) -> None:
        for key in self.tree.selection():
            self.targets.pop(int(key), None)
            values = list(self.tree.item(key, "values"))
            values[-1] = "默认 / 通用 MMD"
            self.tree.item(key, values=values)

    def show_rules(self) -> None:
        path = _resource_root() / "Resources/default-rules.json"
        if path.is_file() and hasattr(os, "startfile"):
            os.startfile(path)

    def build(self) -> None:
        if self.running:
            return
        raw_task = self.task.get().strip()
        if not raw_task:
            messagebox.showerror("缺少任务目录", "请先选择歌曲任务目录。", parent=self)
            return
        try:
            resolved_task = str(Path(raw_task).resolve())
        except OSError as error:
            messagebox.showerror("任务检查失败", str(error), parent=self)
            return
        if self.info is None or self.info["task_dir"] != resolved_task:
            self.inspect()
        if self.info is None:
            return
        slots = [int(x) for x in self.tree.selection()]
        if not slots:
            messagebox.showerror("缺少站位", "请至少选择一个站位。", parent=self)
            return
        try:
            _, build_fn = _load_motion_pipeline()
        except Exception as error:
            messagebox.showerror("缺少运行依赖", str(error), parent=self)
            return
        default_pmx = self.pmx.get().strip() or None
        targets = {s: (self.targets.get(s) or default_pmx) for s in slots}
        args = dict(
            task_dir=raw_task,
            output_root=self.output.get(),
            target_pmx=targets,
            rules_path=self.rules.get() or None,
            slots=slots,
        )
        try:
            self._save_state()
        except OSError as error:
            self.append("设置未能保存: " + str(error))
        self.running = True
        self.start.configure(state="disabled")
        for control in self.controls:
            control.configure(state="disabled")
        self.bar["value"] = 0

        def run() -> None:
            try:
                result = build_fn(**args, progress=lambda **v: self.bus.put(("progress", v)))
                self.bus.put(("done", result))
            except Exception as error:
                self.bus.put(("error", str(error)))

        threading.Thread(target=run, name="scsp-motion-build", daemon=True).start()

    def poll(self) -> None:
        if self._closing or not self.winfo_exists():
            return
        try:
            while True:
                kind, value = self.bus.get_nowait()
                if kind == "progress":
                    self.bar["value"] = value["percent"]
                    self.status.set(value["message"])
                    self.append(value["message"])
                else:
                    self.running = False
                    self.start.configure(state="normal")
                    for control in self.controls:
                        control.configure(state="normal")
                    if kind == "done":
                        self.last_output = value["output_dir"]
                        self.status.set("转换完成: " + str(self.last_output))
                        self.append(self.status.get())
                        messagebox.showinfo("转换完成", str(self.last_output), parent=self)
                    else:
                        self.status.set("转换失败，详情见日志/报告")
                        self.append(value)
                        messagebox.showerror("转换失败", str(value), parent=self)
        except queue.Empty:
            pass
        self._poll_after_id = self.after(120, self.poll)

    def open_output(self) -> None:
        raw = self.last_output or self.output.get()
        p = Path(raw) if raw else Path(self.output.get() or ".")
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        if p.is_dir() and hasattr(os, "startfile"):
            os.startfile(p)

    def stop_polling(self) -> None:
        self._closing = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
