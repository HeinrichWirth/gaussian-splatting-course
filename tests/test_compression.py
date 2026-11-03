import struct
import unittest

import numpy as np

from compression_math import (DEFAULT_STUDY, choose_candidate, memory_budget, pack_motion, pack_static,
                             sample_track, study_settings, unpack_motion, unpack_static)
from reconstruction import small_scene


class CodecTests(unittest.TestCase):
    def test_static_payload_sizes_and_constant_ranges(self):
        scene = small_scene()
        for size,colors,expected in [('f32','rgb',288),('u8','rgb',126),('u8','gray',116)]:
            payload = pack_static(scene,size,100,colors)
            self.assertEqual(len(payload),expected)
            out = unpack_static(payload)
            for source,decoded in zip(scene['splats'],out['splats']):
                np.testing.assert_allclose(decoded['center'],source['center'],atol=.004)
                self.assertTrue(all(s>0 for s in decoded['scales']))
                if colors=='gray':
                    self.assertEqual(len(set(decoded['color'])),1)
            with self.assertRaises(ValueError):
                unpack_static(payload[:-1])
        self.assertEqual(len(unpack_static(pack_static(scene,keep=50))['splats']),3)
        self.assertEqual(memory_budget()['weights'],6552)

    def test_motion_endpoint_short_interval_and_signed_residual(self):
        t = np.linspace(0,1,6)
        track = np.zeros((6,1,3))
        track[:,0,0] = t*(1-t)
        for mode in ('dense','keys','residual'):
            packet = pack_motion(track,mode,3,.002)
            out = unpack_motion(packet)
            np.testing.assert_allclose(out[[0,3,5]],track[[0,3,5]],atol=1e-7)
            if mode=='residual':
                self.assertLessEqual(np.max(abs(out-track)),.001001)
            np.testing.assert_allclose(sample_track(out,.5),(out[2]+out[3])/2)
        with self.assertRaises(ValueError):
            unpack_motion(struct.pack('<4sBBHHHf',b'GSM1',1,0,65535,65535,65535,.1))
        bad = bytearray(pack_static(small_scene(),'f32'))
        struct.pack_into('<f',bad,8+12,-1)
        with self.assertRaises(ValueError):
            unpack_static(bad)

    def test_selection_uses_thresholds_and_can_decline_all(self):
        rows = [dict(name='a',bytes=100,validationPsnr=20,renderMs=5),
                dict(name='b',bytes=300,validationPsnr=21,renderMs=3)]
        for priority,expected in [('size','a'),('quality','b'),('latency','b')]:
            self.assertEqual(choose_candidate(rows,{**DEFAULT_STUDY,'priority':priority})['name'],expected)
        self.assertIsNone(choose_candidate(rows,{**DEFAULT_STUDY,'maxBytes':1}))
        with self.assertRaises(ValueError):
            study_settings({**DEFAULT_STUDY,'repeats':0})
