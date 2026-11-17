"""Input-boundary tests: prevent wrong recipes and downloader path escapes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest import mock
from studio_pipeline import format_pmx_comment, inspect_task

class TaskInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'models').mkdir()
        for name in ('M_Head_28.fbx','M_Dress_2407.fbx'):
            (self.root/'models'/name).touch()
    def tearDown(self):
        self.temp.cleanup()
    def job(self, **values):
        data={'status':'completed', 'dependency_complete':True, 'model_files':['models/M_Head_28.fbx','models/M_Dress_2407.fbx']}
        data.update(values)
        (self.root/'job.json').write_text(json.dumps(data),encoding='utf-8')
    def test_source_identity_drives_character(self):
        self.job()
        self.assertEqual(inspect_task(self.root)['record']['id'],28)
    def test_incomplete_dependency_rejected(self):
        self.job(dependency_complete=False)
        with self.assertRaisesRegex(ValueError,'依赖'): inspect_task(self.root)
    def test_second_head_rejected(self):
        (self.root/'models/M_Head_08.fbx').touch()
        self.job(model_files=['models/M_Head_28.fbx','models/M_Head_08.fbx','models/M_Dress_2407.fbx'])
        with self.assertRaisesRegex(ValueError,'head'): inspect_task(self.root)
    def test_absolute_external_model_rejected(self):
        outside=self.root.parent/(self.root.name+'_outside.fbx')
        outside.touch()
        try:
            self.job(model_files=[str(outside),'models/M_Dress_2407.fbx'])
            with self.assertRaisesRegex(ValueError,'超出'): inspect_task(self.root)
        finally: outside.unlink()
    def test_unknown_character_not_given_other_recipe(self):
        (self.root/'models/M_Head_999.fbx').touch()
        self.job(model_files=['models/M_Head_999.fbx','models/M_Dress_2407.fbx'])
        with self.assertRaisesRegex(ValueError,'角色数据'): inspect_task(self.root)

class BatchQueueTests(unittest.TestCase):
    def run_worker(self, build):
        import queue, threading
        from types import SimpleNamespace
        from studio_gui import StudioGui, BuildJob
        worker=StudioGui.__new__(StudioGui)
        worker._events=queue.Queue()
        worker._load_pipeline=lambda: (SimpleNamespace(build_task=build), None)
        jobs=[BuildJob('a',Path('a'),'', ''),BuildJob('b',Path('b'),'', '')]
        stop=threading.Event()
        return worker,jobs,stop
    def test_failure_continues_next_task(self):
        calls=[]
        def build(task,*args,**kwargs):
            calls.append(task)
            if task=='a': raise RuntimeError('broken task')
            return {'pmx':str(Path('b.pmx').resolve()),'output_dir':str(Path.cwd())}
        worker,jobs,stop=self.run_worker(build)
        worker._build_worker(jobs,Path.cwd(),Path('blender.exe'),stop)
        events=list(worker._events.queue)
        self.assertEqual(calls,['a','b'])
        self.assertEqual(events[-1][1]['succeeded'],1)
        self.assertEqual(events[-1][1]['failed'],1)
    def test_stop_waits_for_current_completion(self):
        calls=[]
        def build(task,*args,**kwargs):
            calls.append(task)
            stop.set()
            return {'pmx':str(Path('a.pmx').resolve()),'output_dir':str(Path.cwd())}
        worker,jobs,stop=self.run_worker(build)
        worker._build_worker(jobs,Path.cwd(),Path('blender.exe'),stop)
        events=list(worker._events.queue)
        self.assertEqual(calls,['a'])
        self.assertTrue(any(e[0]=='task_result' and e[1]=='a' for e in events))
        self.assertTrue(any(e[0]=='task_skipped' and e[1]=='b' for e in events))
        self.assertTrue(events[-1][1]['stopped'])


class CommentAndGuiTests(unittest.TestCase):
    def test_format_pmx_comment_is_verbatim_custom_or_empty(self):
        info = {'name': '杜野凛世', 'romanized_name': 'Morino Rinze', 'character_id': 8}
        self.assertEqual(format_pmx_comment('', info), '')
        self.assertEqual(format_pmx_comment('  \n\r\n ', info), '')
        rendered = format_pmx_comment(
            '版权：Bandai Namco\r\n制作：Tester\r\n适用于 {name} ({romanized}) #{id}\r\n保留：{keep}',
            info,
        )
        self.assertEqual(
            rendered,
            '版权：Bandai Namco\n制作：Tester\n适用于 杜野凛世 (Morino Rinze) #8\n保留：{keep}',
        )

    def test_settings_roundtrip_and_malformed_json(self):
        import studio_gui
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / 'studio_settings.json'
            self.assertEqual(studio_gui.load_studio_settings(settings_file), {})
            studio_gui.save_studio_settings({'default_pmx_comment': '自定义说明'}, settings_file)
            self.assertEqual(
                studio_gui.load_studio_settings(settings_file),
                {'default_pmx_comment': '自定义说明'},
            )
            settings_file.write_text('{bad json', encoding='utf-8')
            self.assertEqual(studio_gui.load_studio_settings(settings_file), {})

    def test_gui_blank_custom_comment_single_selected_and_all_tasks(self):
        import tkinter as tk
        import studio_gui
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / 'studio_settings.json'
            with mock.patch.object(studio_gui, 'STUDIO_SETTINGS_PATH', settings_file):
                root = tk.Tk()
                root.withdraw()
                app = studio_gui.StudioGui(root)
                try:
                    self.assertIsInstance(app._body_paned, tk.PanedWindow)
                    self.assertEqual(app.comment_text.get('1.0', 'end-1c'), '')
                    t1 = studio_gui.StudioTask(key='k1', task_dir=Path(tmp) / 'a', comment='first')
                    t2 = studio_gui.StudioTask(key='k2', task_dir=Path(tmp) / 'b', comment='second')
                    app._tasks['k1'] = t1
                    app._tasks['k2'] = t2
                    app.tree.insert('', 'end', iid='k1', values=app._tree_values(t1))
                    app.tree.insert('', 'end', iid='k2', values=app._tree_values(t2))

                    # Per-task individual configuration when selecting one model
                    app.tree.selection_set('k1')
                    app._on_tree_selection()
                    self.assertEqual(app.comment_text.get('1.0', 'end-1c'), 'first')
                    app.comment_text.delete('1.0', 'end')
                    app.comment_text.insert('1.0', '模型1单独说明')
                    app._flush_comment_editor()
                    self.assertEqual(t1.comment, '模型1单独说明')
                    self.assertEqual(t2.comment, 'second')

                    # Apply to selected task(s) only
                    app.comment_text.delete('1.0', 'end')
                    app.comment_text.insert('1.0', '仅应用给k1')
                    app._apply_comment_to_selected()
                    self.assertEqual(t1.comment, '仅应用给k1')
                    self.assertEqual(t2.comment, 'second')

                    # Apply to all & persist default
                    app.comment_text.delete('1.0', 'end')
                    app.comment_text.insert('1.0', '全部统一说明 {name}')
                    app._apply_comment_to_all()
                    self.assertEqual(t1.comment, '全部统一说明 {name}')
                    self.assertEqual(t2.comment, '全部统一说明 {name}')
                    app._save_default_comment()
                    self.assertEqual(
                        studio_gui.load_studio_settings(settings_file)['default_pmx_comment'],
                        '全部统一说明 {name}',
                    )
                finally:
                    app._closing = True
                    app._inspect_jobs.put(None)
                    root.destroy()

    def test_comment_box_and_actions_visible_at_default_geometry(self) -> None:
        import tkinter as tk
        import studio_gui

        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / 'studio_settings.json'
            with mock.patch.object(studio_gui, 'STUDIO_SETTINGS_PATH', settings_file):
                root = tk.Tk()
                app = studio_gui.StudioGui(root)
                try:
                    for geometry in ('+-10000+-10000', '980x720+-10000+-10000'):
                        root.geometry(geometry)
                        root.update()
                        app._on_body_mapped()
                        root.update()
                        comment_shell = app.comment_text.master.master
                        details_pad = comment_shell.master
                        actions_frame = app.save_default_comment_button.master
                        min_comment_h = 40 if geometry.startswith('980x720') else 90
                        self.assertGreaterEqual(app.comment_text.winfo_height(), min_comment_h, geometry)
                        self.assertGreaterEqual(app.comment_text.winfo_width(), 180, geometry)
                        self.assertGreater(comment_shell.winfo_y(), 0, geometry)
                        self.assertLessEqual(
                            actions_frame.winfo_x() + actions_frame.winfo_width(),
                            details_pad.winfo_width(),
                            geometry,
                        )
                        self.assertLessEqual(
                            actions_frame.winfo_y() + actions_frame.winfo_height(),
                            details_pad.winfo_height(),
                            geometry,
                        )
                finally:
                    app._closing = True
                    app._inspect_jobs.put(None)
                    root.destroy()

    def test_skin_processing_classification_and_tints(self) -> None:
        import skin_processing

        # Izumi Mei (character 19) neutral bodyskin (255, 239, 226) -> (230, 171, 160).
        tint = skin_processing.skin_tint('#663735')
        rgb_8bit = tuple(
            round(min(1.0, max(0.0, c * t)) * 255)
            for c, t in zip((1.0, 239 / 255.0, 226 / 255.0), tint)
        )
        self.assertEqual(rgb_8bit, (230, 171, 160))

        # Palette validation gracefully skips incomplete records without inventing values.
        palette, warnings = skin_processing._extract_skin_palette({'id': 999})
        self.assertEqual(palette, {})
        self.assertEqual(len(warnings), 2)

        # Classification isolates M_face & body skins while excluding mouth/teeth/hair.
        self.assertEqual(
            skin_processing.classify_skin_material('M_face', '19_face_00_col.png', 'head'),
            'face',
        )
        self.assertIsNone(
            skin_processing.classify_skin_material('M_mouth_inner', '19_face_00_col.png', 'head')
        )
        self.assertIsNone(
            skin_processing.classify_skin_material('M_teeth_tongue', '19_face_00_col.png', 'head')
        )
        self.assertEqual(
            skin_processing.classify_skin_material(
                'M_0045_18NLRS_skin_00_bloom', '00_bodyskin_00_col.png', 'dress'
            ),
            'body',
        )
        self.assertIsNone(
            skin_processing.classify_skin_material('M_hair_00', '19_hair_00_col.png', 'hair')
        )

    def test_skin_resolve_face_shade_path_prefers_texture_directory(self) -> None:
        import skin_processing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head_dir = root / 'head_08'
            acce_dir = root / 'acce_08'
            head_dir.mkdir()
            acce_dir.mkdir()
            (head_dir / '08_face_00_col.png').write_bytes(b'x')
            (head_dir / '08_face_sd.png').write_bytes(b'sd-head')
            (acce_dir / '08_face_0000_col.png').write_bytes(b'x')
            (acce_dir / '08_face_sd.png').write_bytes(b'sd-acce')

            match, warning = skin_processing.resolve_face_shade_path(
                acce_dir / '08_face_0000_col.png', root
            )
            self.assertIsNone(warning)
            self.assertEqual(match, acce_dir / '08_face_sd.png')

    def test_morph_config_removed_and_generated_cp932_safe(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / 'Resources' / 'unity_morph_names.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        self.assertEqual(
            set(cfg.get('removed_sources', [])),
            {'mouth_open', 'eyelashOP_L_on', 'eyelashOP_R_on', 'tongue_up'},
        )
        combo_names = {c['name'] for c in cfg.get('combinations', [])}
        self.assertIn('個性', combo_names)
        for combo in cfg.get('combinations', []):
            self.assertLessEqual(len(combo['name'].encode('cp932')), cfg['name_max_bytes'], combo['name'])
            self.assertIn(combo['panel'], {1, 2, 3}, combo['name'])
            self.assertTrue(combo['english'])
        for name, meta in cfg.get('generated_morphs', {}).items():
            encoded = name.encode('cp932')
            self.assertLessEqual(len(encoded), cfg['name_max_bytes'], name)
            self.assertIn(meta['panel'], {1, 2, 3}, name)
            self.assertTrue(meta['english'])

    def test_pmx_io_binary_roundtrip_and_weights(self) -> None:
        tools_dir = Path(__file__).resolve().parent
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        import pmx_io

        def make_bone(name, name_e, loc, parent, conn):
            b = pmx_io.Bone()
            b.name, b.name_e, b.location, b.parent, b.displayConnection = name, name_e, loc, parent, conn
            return b

        def make_vert(co, normal, uv, w_type, bones, weights):
            v = pmx_io.Vertex()
            v.co, v.normal, v.uv = co, normal, uv
            v.weight = pmx_io.BoneWeight()
            v.weight.type, v.weight.bones, v.weight.weights = w_type, bones, weights
            return v

        m = pmx_io.Model()
        m.name = 'テストモデル'
        m.name_e = 'TestModel'
        m.comment = 'line1\nline2'
        m.comment_e = 'eng'
        m.header = pmx_io.Header()
        m.textures = [pmx_io.Texture('tex/body.png')]
        b0 = make_bone('全ての親', 'root', [0.0, 0.0, 0.0], -1, 1)
        b1 = make_bone('足IK', 'legIK', [1.0, 0.5, 0.0], 0, [0.0, -1.0, 0.0])
        b1.isIK, b1.target, b1.loopCount, b1.rotationConstraint = True, 0, 40, 2.0
        lk = pmx_io.IKLink()
        lk.target, lk.maximumAngle, lk.minimumAngle = 0, [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]
        b1.ik_links = [lk]
        m.bones = [b0, b1]
        m.vertices = [
            make_vert([0.0, 1.0, 2.0], [0.0, 0.0, 1.0], [0.25, 0.75], pmx_io.BoneWeight.BDEF1, [0], [1.0]),
            make_vert([1.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0.5, 0.5], pmx_io.BoneWeight.BDEF2, [0, 1], [0.4]),
            make_vert(
                [1.0, 2.0, 2.0], [1.0, 0.0, 0.0], [0.75, 0.25], pmx_io.BoneWeight.SDEF, [0, 1],
                pmx_io.SDEFWeight(weight=0.3, c=(0.5, 1.5, 2.0), r0=(0.0, 1.0, 2.0), r1=(1.0, 2.0, 2.0)),
            ),
        ]
        m.faces = [(0, 1, 2)]
        mat = pmx_io.Material()
        mat.name, mat.name_e, mat.texture, mat.vertex_count = 'Mat0', 'Mat0E', 0, 3
        m.materials = [mat]
        vm = pmx_io.VertexMorph('あ', 'a', 3)
        vm.offsets.append(pmx_io.VertexMorphOffset(0, [0.0, 0.1, -0.05]))
        gm = pmx_io.GroupMorph('個性', 'Unique', 1)
        gm.offsets.append(pmx_io.MorphData(0, 1.0))
        m.morphs = [vm, gm]
        d0 = pmx_io.Display()
        d0.name, d0.name_e, d0.isSpecial, d0.data = 'Root', 'Root', True, [(0, 0)]
        d1 = pmx_io.Display()
        d1.name, d1.name_e, d1.isSpecial, d1.data = '表情', 'Exp', False, [(1, 0)]
        m.display = [d0, d1]
        rb = pmx_io.Rigid()
        rb.name, rb.name_e, rb.bone = 'RB0', 'RB0', 0
        m.rigids = [rb]
        jn = pmx_io.Joint()
        jn.name, jn.name_e, jn.src_rigid, jn.dest_rigid = 'J0', 'J0', 0, 0
        m.joints = [jn]

        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / 'out1.pmx'
            p2 = Path(tmp) / 'out2.pmx'
            pmx_io.save(p1, m)
            reloaded = pmx_io.load(p1)
            pmx_io.save(p2, reloaded)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            self.assertEqual(len(reloaded.morphs), 2)
            self.assertEqual(reloaded.vertices[2].weight.type, pmx_io.BoneWeight.SDEF)
            self.assertAlmostEqual(reloaded.vertices[2].weight.weights.c[1], 1.5, places=5)

    def test_opensource_io_defaults_and_artifacts_present(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        defaults = json.loads((repo_root / 'Resources' / 'pipeline-defaults.json').read_text(encoding='utf-8'))
        self.assertEqual(defaults.get('importer'), 'ufbx')
        self.assertTrue((repo_root / 'Tools' / 'ufbx_bridge.dll').is_file())
        self.assertTrue((repo_root / 'Tools' / 'ufbx_importer.py').is_file())
        self.assertTrue((repo_root / 'Tools' / 'pmx_io.py').is_file())

    def test_blender_version_probing_and_gui_badge(self) -> None:
        import tkinter as tk
        import studio_gui
        import studio_pipeline

        self.assertEqual(studio_pipeline.parse_blender_version('Blender 4.2.23 LTS\n'), (4, 2, 23))
        self.assertEqual(studio_pipeline.parse_blender_version('Blender 5.3.0 Release Candidate\n'), (5, 3, 0))
        self.assertIsNone(studio_pipeline.parse_blender_version('not blender'))

        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = Path(tmp) / 'blender.exe'
            fake_exe.write_bytes(b'MZ')
            studio_pipeline._BLENDER_PROBE_CACHE.clear()

            with mock.patch.object(
                studio_pipeline.subprocess,
                'run',
                return_value=mock.Mock(returncode=0, stdout='Blender 4.2.23 LTS\n', stderr=''),
            ):
                p42 = studio_pipeline.probe_blender_executable(fake_exe)
                self.assertTrue(p42['ok'])
                self.assertTrue(p42['supported'])
                self.assertEqual(p42['version'], (4, 2, 23))
                self.assertEqual(studio_pipeline.validate_blender_executable(fake_exe), fake_exe.resolve())

            studio_pipeline._BLENDER_PROBE_CACHE.clear()
            with mock.patch.object(
                studio_pipeline.subprocess,
                'run',
                return_value=mock.Mock(returncode=0, stdout='Blender 4.0.2\n', stderr=''),
            ):
                p40 = studio_pipeline.probe_blender_executable(fake_exe)
                self.assertTrue(p40['ok'])
                self.assertFalse(p40['supported'])
                with self.assertRaisesRegex(ValueError, '4.0.2'):
                    studio_pipeline.validate_blender_executable(fake_exe)

            settings_file = Path(tmp) / 'studio_settings.json'
            with mock.patch.object(studio_gui, 'STUDIO_SETTINGS_PATH', settings_file):
                root = tk.Tk()
                root.withdraw()
                app = studio_gui.StudioGui(root)
                try:
                    with mock.patch.object(
                        studio_gui,
                        'probe_blender_executable',
                        return_value={'ok': True, 'supported': True, 'version': (4, 2, 23), 'version_text': '4.2.23', 'error': None},
                    ):
                        app._refresh_env_badges()
                        labels = [child.cget('text') for child in app.env_badge_frame.winfo_children()]
                        self.assertIn('● Blender 4.2', labels)

                    with mock.patch.object(
                        studio_gui,
                        'probe_blender_executable',
                        return_value={'ok': True, 'supported': False, 'version': (4, 0, 2), 'version_text': '4.0.2', 'error': '需要 >= 4.1'},
                    ):
                        app._refresh_env_badges()
                        labels = [child.cget('text') for child in app.env_badge_frame.winfo_children()]
                        self.assertIn('○ Blender 4.0 (<4.1)', labels)
                finally:
                    app._closing = True
                    app._inspect_jobs.put(None)
                    root.destroy()

    def test_pe_core_writes_portable_relative_texture_paths_and_audit_accepts_them(self) -> None:
        import subprocess
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / 'Tools'))
        import pmx_io

        required = [
            '全ての親', 'センター', 'グルーブ',
            '左足ＩＫ', '右足ＩＫ', '左つま先ＩＫ', '右つま先ＩＫ', '左足IK親', '右足IK親',
            '左腕捩', '右腕捩', '左手捩', '右手捩', '左肩P', '右肩P', '左肩C', '右肩C',
            '下半身', '上半身', '頭', '左目', '右目',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / '_work'
            work.mkdir()
            src_tex = work / 'body.png'
            src_tex.write_bytes(b'\x89PNG\r\n\x1a\n')
            in_pmx = work / '03-semi.pmx'
            out_pmx = root / 'FinalModel.pmx'
            report_path = work / 'audit.json'

            m = pmx_io.Model()
            m.header = pmx_io.Header()
            m.name = 'Test'
            m.textures = [pmx_io.Texture(path=str(src_tex))]
            for i, bname in enumerate(required):
                b = pmx_io.Bone()
                b.name, b.name_e = bname, bname
                b.parent = -1 if i == 0 else 0
                b.location = [float(i), 1.0, 0.0]
                if 'ＩＫ' in bname:
                    b.isIK = True
                    b.target = 0
                    lk = pmx_io.IKLink()
                    lk.target = 0
                    b.ik_links = [lk]
                m.bones.append(b)
            for co in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
                v = pmx_io.Vertex()
                v.co = co
                v.weight = pmx_io.BoneWeight()
                v.weight.type = pmx_io.BoneWeight.BDEF1
                v.weight.bones = [0]
                m.vertices.append(v)
            m.faces = [(0, 1, 2)]
            mat = pmx_io.Material()
            mat.name, mat.texture, mat.vertex_count = 'Mat0', 0, 3
            m.materials = [mat]
            pmx_io.save(in_pmx, m)

            subprocess.run(
                [sys.executable, str(repo_root / 'Tools' / 'pe_core.py'), '--', '--input', str(in_pmx), '--output', str(out_pmx), '--phase', 'finalize'],
                check=True,
                capture_output=True,
                text=True,
            )
            final_model = pmx_io.load(out_pmx)
            self.assertEqual([t.path for t in final_model.textures], ['textures\\00_body.png'])
            self.assertTrue((root / 'textures' / '00_body.png').is_file())

            subprocess.run(
                [sys.executable, str(repo_root / 'Tools' / 'audit_pe_result.py'), '--', '--input', str(in_pmx), '--output', str(out_pmx), '--report', str(report_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            audit = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(audit['textures_missing'], [])
            self.assertEqual(audit['textures_non_relative'], [])

    def test_discover_default_blender_ignores_inaccessible_drive_oserror(self) -> None:
        import studio_pipeline

        orig_is_dir = Path.is_dir
        orig_is_file = Path.is_file

        def fake_is_dir(self_path: Path) -> bool:
            if str(self_path).upper().startswith('D:'):
                raise OSError(22, '参数错误', str(self_path))
            return False

        def fake_is_file(self_path: Path) -> bool:
            if str(self_path).upper().startswith('D:'):
                raise OSError(22, '参数错误', str(self_path))
            return False

        with mock.patch.dict(studio_pipeline.os.environ, {'SCSP_MMD_BLENDER': ''}, clear=False), \
             mock.patch.object(studio_pipeline.shutil, 'which', return_value=None), \
             mock.patch.object(Path, 'is_dir', autospec=True, side_effect=fake_is_dir), \
             mock.patch.object(Path, 'is_file', autospec=True, side_effect=fake_is_file):
            res = studio_pipeline.discover_default_blender()
            self.assertEqual(res, Path('blender'))


if __name__=='__main__':unittest.main()

