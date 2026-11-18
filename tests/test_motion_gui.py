"""Smoke checks for top tab switcher and embedded MotionTab."""
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import studio_gui
from motion_tab import MotionTab


def _collect_label_texts(widget: tk.Misc) -> list[str]:
    texts: list[str] = []
    for child in widget.winfo_children():
        try:
            text = child.cget("text")
            if isinstance(text, str) and text:
                texts.append(text)
        except tk.TclError:
            pass
        texts.extend(_collect_label_texts(child))
    return texts


class MotionGuiIntegrationTests(unittest.TestCase):
    def test_top_tabs_switch_and_motion_tab_preserves_workflow_without_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "studio_settings.json"
            with patch.object(studio_gui, "STUDIO_SETTINGS_PATH", settings_file):
                try:
                    root = tk.Tk()
                except tk.TclError:
                    self.skipTest("Tk display unavailable")
                root.withdraw()
                app = studio_gui.StudioGui(root)
                try:
                    root.geometry("1220x860")
                    root.update()
                    self.assertTrue(app.pmx_tab_button.winfo_exists())
                    self.assertTrue(app.motion_tab_button.winfo_exists())

                    app._switch_tab("motion")
                    root.update_idletasks()
                    self.assertEqual(app._active_tab, "motion")
                    self.assertEqual(app._motion_page.winfo_manager(), "pack")
                    self.assertEqual(app._pmx_page.winfo_manager(), "")
                    self.assertEqual(app.motion_tab_button.cget("style"), "TabActive.TButton")
                    self.assertEqual(app.pmx_tab_button.cget("style"), "TabInactive.TButton")

                    texts = "\n".join(_collect_label_texts(app.motion_tab))
                    for unwanted in (
                        "骨骼适配",
                        "选中的站位会导出",
                        "标准 30 FPS",
                        "留空=通用导出",
                        "PoseWorker",
                    ):
                        self.assertNotIn(unwanted, texts)

                    fake = dict(
                        task_dir=str(Path(tmp).resolve()),
                        song="test",
                        mouths=["a", "b"],
                        warnings=[],
                        slots=[
                            dict(slot=1, character_id=14, duration=2),
                            dict(slot=2, character_id=12, duration=2),
                        ],
                    )
                    app.motion_tab.task.set(tmp)
                    with patch("motion_tab.inspect_task", return_value=fake):
                        app.motion_tab.inspect()
                    self.assertEqual(app.motion_tab.tree.selection(), ("1", "2"))
                    self.assertEqual(app.motion_tab.tree.item("1", "values")[1], "14")
                    self.assertEqual(
                        app.motion_tab.status.get(),
                        "test · 2 个站位 · 口型 2 份",
                    )

                    captured: dict[str, object] = {}

                    def fake_build(**kwargs: object) -> dict[str, str]:
                        captured.update(kwargs)
                        return {"output_dir": tmp}

                    with patch("motion_tab.build_task", side_effect=fake_build), patch(
                        "motion_tab.threading.Thread"
                    ) as th, patch("motion_tab.messagebox.showerror") as err:
                        def run_thread(*_a, target, daemon=None, **_kw):
                            self.assertTrue(daemon)
                            return type("T", (), {"start": lambda self: target()})()
                        th.side_effect = run_thread
                        app.motion_tab.pmx.set("")
                        app.motion_tab.output.set(tmp)
                        app.motion_tab.build()

                    err.assert_not_called()
                    self.assertEqual(captured["target_pmx"], {1: None, 2: None})
                    self.assertNotIn("backend_path", captured)
                    saved = studio_gui.load_studio_settings(settings_file)
                    self.assertEqual(saved.get("motion_task"), tmp)
                    self.assertEqual(saved.get("motion_output"), tmp)

                    # Empty task directory should report cleanly instead of inspecting CWD.
                    with patch("motion_tab.messagebox.showinfo"):
                        app.motion_tab.poll()
                    self.assertFalse(app.motion_tab.running)
                    app.motion_tab.task.set("")
                    with patch("motion_tab.messagebox.showerror") as empty_err:
                        app.motion_tab.build()
                    empty_err.assert_called_once()

                    app._switch_tab("pmx")
                    root.update_idletasks()
                    self.assertEqual(app._pmx_page.winfo_manager(), "pack")
                    self.assertEqual(app._motion_page.winfo_manager(), "")
                    self.assertEqual(app.pmx_tab_button.cget("style"), "TabActive.TButton")
                    self.assertEqual(app.motion_tab_button.cget("style"), "TabInactive.TButton")
                finally:
                    app._closing = True
                    app.motion_tab.stop_polling()
                    app._inspect_jobs.put(None)
                    root.destroy()

    def test_standalone_motion_tab_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = tk.Tk()
            except tk.TclError:
                self.skipTest("Tk display unavailable")
            root.withdraw()
            try:
                tab = MotionTab(root, default_output_root=Path(tmp))
                tab.pack(fill="both", expand=True)
                self.assertTrue(tab.start.winfo_exists())
            finally:
                tab.stop_polling()
                root.destroy()


if __name__ == "__main__":
    unittest.main()
