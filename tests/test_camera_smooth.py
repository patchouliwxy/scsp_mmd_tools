import unittest
import numpy as np
from scsp_motion.camera_smooth import slopes,sample_series,encode_camera


def segment(a,b,cut=False,transition=0.,reference=(0.,0.,0.)):
    return dict(start_vmd_frame=a,end_vmd_frame=b,hard_cut=cut,
                transition_seconds=transition,zoom_reference=list(reference))


def interpolate_key(a,b,u):
    """Independent decoder of the written camera interpolation byte layout."""
    def weight(channel):
        x1,x2,y1,y2=np.frombuffer(b['interpolation'][channel*4:channel*4+4],dtype=np.uint8)/127.
        lo,hi=0.,1.
        for _ in range(45):
            z=(lo+hi)/2;x=3*(1-z)**2*z*x1+3*(1-z)*z*z*x2+z**3
            if x<u:lo=z
            else:hi=z
        z=(lo+hi)/2
        return 3*(1-z)**2*z*y1+3*(1-z)*z*z*y2+z**3
    p=np.array([a['position'][i]+(b['position'][i]-a['position'][i])*weight(i) for i in range(3)])
    return p,a['fov']+(b['fov']-a['fov'])*weight(5)


class CameraSmoothTests(unittest.TestCase):
    def test_shape_preserving_sampling_no_overshoot_or_pause_in_linear_move(self):
        t=np.linspace(0,1,1001)
        np.testing.assert_allclose(sample_series([0.,1.,2.],t),2*t,atol=1e-12)
        result=sample_series([0.,0.,1.,1.],t)
        self.assertGreaterEqual(result.min(),0);self.assertLessEqual(result.max(),1)
        self.assertTrue(np.all(np.diff(result)>=0))

    def test_position_tangents_reduce_velocity_jumps(self):
        x=np.arange(8,dtype=float)**2*.1;p=np.c_[x,np.zeros((8,2))]
        keys,_,_=encode_camera(p,np.zeros_like(p),np.full(8,30.),[segment(0,7)],compensate=False)
        jumps=[]
        for i in range(1,7):
            prev=keys[i]['interpolation'][:4];nxt=keys[i+1]['interpolation'][:4]
            left=(x[i]-x[i-1])*(127-prev[3])/(127-prev[1])
            right=(x[i+1]-x[i])*nxt[2]/nxt[0]
            jumps.append(abs(left-right))
        self.assertLess(max(jumps),.025)
        self.assertGreater(np.max(abs(np.diff(np.diff(x)))),.19)

    def test_fractional_zoom_preserves_projection_at_reference_plane(self):
        n=121;p=np.tile([0.,0.,-10.],(n,1));r=np.zeros_like(p);f=np.linspace(20.2,22.6,n)
        original=p.copy();keys,centers,report=encode_camera(p,r,f,[segment(0,n-1)])
        np.testing.assert_array_equal(p,original)
        for i,k in enumerate(keys):
            expected=1/(10*np.tan(np.deg2rad(f[i])/2))
            actual=1/(-centers[i,2]*np.tan(np.deg2rad(k['fov'])/2))
            self.assertAlmostEqual(expected,actual,places=12)
        self.assertEqual(report['zoom_compensation']['frames_compensated'],n)

    def test_subframe_zoom_steps_reduced_against_uncompensated_rounding(self):
        n=121;p=np.tile([0.,0.,-10.],(n,1));f=np.linspace(20.2,22.6,n)
        new,_,_=encode_camera(p,np.zeros_like(p),f,[segment(0,n-1)])
        old,_,_=encode_camera(p,np.zeros_like(p),f,[segment(0,n-1)],compensate=False)
        errors=[[],[]]
        for i in range(n-1):
            for u in (.25,.5,.75):
                expected=1/(10*np.tan(np.deg2rad(f[i]*(1-u)+f[i+1]*u)/2))
                for dst,keys in zip(errors,(new,old)):
                    location,angle=interpolate_key(keys[i],keys[i+1],u)
                    value=1/(-location[2]*np.tan(np.deg2rad(angle)/2))
                    dst.append(abs(value/expected-1))
        self.assertLess(max(errors[0]),max(errors[1])/20)

    def test_hard_cut_position_and_integer_zoom_preserved(self):
        p=np.r_[np.tile([0.,0.,-10.],(30,1)),np.tile([10.,2.,-5.],(30,1))]
        f=np.r_[np.full(30,30.),np.full(30,45.)]
        keys,centers,_=encode_camera(p,np.zeros_like(p),f,[segment(0,29),segment(30,59,True)])
        np.testing.assert_allclose(centers,p)
        self.assertEqual(keys[30]['interpolation'],bytes([127,127,0,0]*6))

    def test_fast_sixty_fps_zoom_midpoint_is_not_replaced_by_endpoint_average(self):
        f=np.array([22.89,25.78,34.84,37.21,37.29]);p=np.tile([0.,0.,-10.],(5,1))
        keys,_,_=encode_camera(p,np.zeros_like(p),f,[segment(0,2)],source_fps=60)
        for i in range(2):
            location,angle=interpolate_key(keys[i],keys[i+1],.5)
            actual=1/(-location[2]*np.tan(np.deg2rad(angle)/2))
            expected=1/(10*np.tan(np.deg2rad(f[2*i+1])/2))
            self.assertLess(abs(actual/expected-1),.01)

    def test_soft_transition_blends_only_incoming_shot(self):
        p=np.r_[np.tile([0.,0.,-10.],(30,1)),np.tile([10.,0.,-10.],(30,1))]
        original=p.copy()
        keys,centers,report=encode_camera(p,np.zeros_like(p),np.full(60,30.),
            [segment(0,29),segment(30,59,transition=.5)],compensate=False)
        np.testing.assert_array_equal(centers[:30],p[:30])
        self.assertEqual(centers[30,0],0)
        np.testing.assert_allclose(centers[45:],p[45:])
        self.assertLess(np.max(np.diff(centers[:,0])),1.3)
        np.testing.assert_array_equal(p,original)
        self.assertEqual(report['transitions'][0]['end_frame'],45)

    def test_reference_behind_camera_does_not_shift_view(self):
        p=np.tile([0.,0.,10.],(4,1));keys,centers,report=encode_camera(p,np.zeros_like(p),np.full(4,30.2),[segment(0,3)])
        np.testing.assert_array_equal(centers,p)
        self.assertEqual(report['zoom_compensation']['frames_without_forward_reference'],4)

    def test_sixty_fps_odd_boundary_blends_from_previous_output_frame(self):
        # The incoming shot starts halfway between VMD frames 29 and 30.
        p=np.r_[np.tile([0.,0.,-10.],(59,1)),np.tile([10.,0.,-10.],(60,1))]
        keys,centers,_=encode_camera(p,np.zeros_like(p),np.full(119,30.),
            [segment(0,29),segment(30,59,transition=.5)],source_fps=60,compensate=False)
        self.assertEqual(centers[29,0],0)
        self.assertEqual(centers[30,0],0)
        self.assertEqual(centers[45,0],10)

    def test_dynamic_zoom_references_keep_projection_along_moving_target(self):
        # Camera and target both advance by +30 in Z so true depth stays 10 everywhere,
        # whereas static seg['zoom_reference']=[0,0,0] would invert behind the camera after z>0.
        z=np.linspace(-5.,25.,9)
        p=np.column_stack([np.zeros(9),np.zeros(9),z])
        zrefs=np.column_stack([np.zeros(9),np.zeros(9),z+10.])
        f=np.full(9,30.4)
        keys,centers,report=encode_camera(p,np.zeros_like(p),f,[segment(0,4)],source_fps=60,zoom_references=zrefs)
        self.assertEqual(report['zoom_compensation']['frames_compensated'],5)
        self.assertEqual(report['zoom_compensation']['frames_without_forward_reference'],0)
        for i in range(4):
            location,angle=interpolate_key(keys[i],keys[i+1],.5)
            actual=1/((zrefs[2*i+1,2]-location[2])*np.tan(np.deg2rad(angle)/2))
            expected=1/(10*np.tan(np.deg2rad(30.4)/2))
            self.assertLess(abs(actual/expected-1),.01)

    def test_sparse_reduce_merges_smooth_span_and_preserves_hard_cut(self):
        t0=np.linspace(0.,1.,59)
        t1=np.linspace(0.,1.,60)
        p=np.r_[np.column_stack([20.*t0,np.zeros(59),np.full(59,-10.)]),
                np.column_stack([np.full(60,5.),10.*t1,np.full(60,-15.)])]
        f=np.r_[np.linspace(30.,20.,59),np.full(60,35.)]
        keys,_,report=encode_camera(p,np.zeros_like(p),f,[segment(0,29),segment(30,59,cut=True)],
                                    compensate=False,source_fps=60,sparse_reduce=True)
        self.assertTrue(report['sparse_reduced'])
        self.assertLess(len(keys),20)
        cut_key=next(k for k in keys if k['frame']==30)
        self.assertEqual(cut_key['interpolation'],bytes([127,127,0,0]*6))

    def test_max_shift_pmx_caps_telephoto_zoom_compensation(self):
        p=np.tile([0.,0.,-100.],(8,1))
        f=np.full(8,10.4)
        _,_,rep_uncapped=encode_camera(p,np.zeros_like(p),f,[segment(0,7)],compensate=True)
        _,_,rep_capped=encode_camera(p,np.zeros_like(p),f,[segment(0,7)],compensate=True,max_shift_pmx=0.35)
        self.assertGreater(rep_uncapped['zoom_compensation']['max_shift_pmx'],3.0)
        self.assertLessEqual(rep_capped['zoom_compensation']['max_shift_pmx'],0.35+1e-9)

    def test_hard_cut_wraps_each_block_start_within_principal_period(self):
        p=np.tile([0.,0.,-10.],(8,1))
        r=np.r_[np.tile([-2*np.pi-0.1,0.,0.],(4,1)),np.tile([2*np.pi+0.1,0.,0.],(4,1))]
        keys,_,_=encode_camera(p,r,np.full(8,30.),[segment(0,3),segment(4,7,cut=True)],compensate=False)
        self.assertLessEqual(abs(keys[0]['rotation'][0]),np.pi)
        self.assertLessEqual(abs(keys[4]['rotation'][0]),np.pi)
        self.assertLess(abs(keys[4]['rotation'][0]-keys[3]['rotation'][0]),1.0)

    def test_long_soft_transition_damps_outgoing_velocity(self):
        # Fast outgoing shot (+5/frame at 60fps = 10/VMD-frame) followed by 2.0s transition into static shot at x=0.
        p=np.r_[np.column_stack([np.arange(19)*5.0,np.zeros(19),np.full(19,-10.)]),
                np.zeros((120,3))+[0.,0.,-10.]]
        keys,centers,_=encode_camera(p,np.zeros_like(p),np.full(139,30.),
            [segment(0,9),segment(10,69,transition=2.0)],source_fps=60,compensate=False)
        # Outgoing end is x=90 at frame 9; damped extrapolation over tau=0.15s stays bounded instead of flying to ~200+.
        self.assertLess(np.max(centers[10:,0]),140.0)

    def test_sparse_reduce_emits_only_monotonic_bezier_handles(self):
        t=np.linspace(0.,1.,121)
        p=np.column_stack([20.*np.sin(2.5*np.pi*t),5.*t**2,-10.+3.*np.cos(2*np.pi*t)])
        r=np.column_stack([0.2*np.sin(3*np.pi*t),0.4*t,0.1*np.cos(2*np.pi*t)])
        f=30.+8.*np.sin(np.pi*t)
        keys,_,_=encode_camera(p,r,f,[segment(0,30),segment(31,60,cut=True)],
                               compensate=False,source_fps=60,sparse_reduce=True)
        step=bytes([127,127,0,0]*6)
        for k in keys:
            ip=k['interpolation']
            if ip==step:continue
            for c in range(6):
                x1,x2,y1,y2=ip[c*4:c*4+4]
                self.assertLessEqual(x1,x2,f'non-monotonic X handle on frame {k["frame"]} ch {c}')
                self.assertLessEqual(y1,y2,f'non-monotonic Y handle on frame {k["frame"]} ch {c}')

    def test_interrupted_soft_transition_preserves_configured_duration_rate(self):
        # Shot 1 has a 2.0s transition (60 VMD frames) but is cut short after 15 VMD frames.
        p=np.r_[np.zeros((30,3)),np.tile([20.,0.,-10.],(15,1)),np.tile([20.,0.,-10.],(30,1))]
        _,centers,_=encode_camera(p,np.zeros_like(p),np.full(75,30.),
            [segment(0,29),segment(30,44,transition=2.0),segment(45,74,cut=True)],
            source_fps=30,compensate=False)
        # After 14 frames of a 60-frame smoothstep (u=14/60 < 0.25), blend should still be well below 0.25 (<5.0).
        self.assertLess(centers[44,0],5.0)

    def test_nonuniform_time_grid_pchip_avoids_rate_change_overshoot(self):
        # Linear motion sampled at variable time steps (4x spacing change mid-clip) must stay linear in time.
        gx=np.r_[np.linspace(0.,0.2,9),np.linspace(0.28,1.0,10)]
        vals=10.0*gx
        tq=np.linspace(0.,1.,101)
        res=sample_series(vals,tq,grid_x=gx)
        np.testing.assert_allclose(res,10.0*tq,atol=1e-12)
