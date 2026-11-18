import copy
import unittest
import numpy as np
from scsp_motion.camera import _merge_easing_duplicates
from scsp_motion.curves import sample


class CameraEasingTests(unittest.TestCase):
    def test_repeated_joins_keep_incoming_and_outgoing_tangents(self):
        curve=dict(m_Curve=[dict(time=0.,value=0.,outSlope=1.),
            dict(time=.5,value=.5,inSlope=1.,outSlope=99.,weightedMode=1,inWeight=.25),
            dict(time=.5,value=.5,inSlope=99.,outSlope=1.,weightedMode=2,outWeight=.25),
            dict(time=1.,value=1.,inSlope=1.)])
        original=copy.deepcopy(curve);merged,count=_merge_easing_duplicates(curve)
        self.assertEqual(curve,original)
        self.assertEqual(count,1)
        self.assertEqual(merged['m_Curve'][1]['inSlope'],1.)
        self.assertEqual(merged['m_Curve'][1]['outSlope'],1.)
        self.assertEqual(merged['m_Curve'][1]['weightedMode'],3)
        times=np.linspace(0,1,101)
        np.testing.assert_allclose(sample(merged,times),times,atol=1e-9)

    def test_conflicting_duplicate_values_are_not_guessed(self):
        with self.assertRaisesRegex(ValueError,'同一时间存在不同值'):
            _merge_easing_duplicates(dict(m_Curve=[dict(time=0,value=0),dict(time=0,value=1)]))

    def test_descending_times_still_rejected(self):
        curve,_=_merge_easing_duplicates(dict(m_Curve=[dict(time=1,value=1),dict(time=0,value=0)]))
        with self.assertRaisesRegex(ValueError,'strictly increasing'):sample(curve,[.5])
