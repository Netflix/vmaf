__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import re
import subprocess
import unittest

import config

def read_log(log_filename, type):
    scores = []
    idx = 0
    with open(log_filename, 'rt') as log_file:
        for line in log_file.readlines():
            mo = re.match("{type}: ([0-9]+) ([0-9.-]+)".format(type=type), line)
            if mo:
                cur_idx = int(mo.group(1))
                assert cur_idx == idx
                scores.append(float(mo.group(2)))
                idx += 1
    score = sum(scores) / float(len(scores))
    return score, scores

class FeatureTest(unittest.TestCase):

    VMAF = config.ROOT + "/feature/vmaf"
    PSNR = config.ROOT + "/feature/psnr"
    MOMENT = config.ROOT + "/feature/moment"
    LOG_FILENAME = config.ROOT + "/workspace/log"
    REF_YUV = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
    DIS_YUV = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
    YUV_FMT = "yuv420p"
    YUV_WIDTH = 576
    YUV_HEIGHT = 324

    def setUp(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def tearDown(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def test_adm(self):
        print 'test adm...'
        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666)

    def test_ansnr(self):
        print 'test ansnr...'
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333)

    def test_motion(self):
        print 'test motion...'
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667)

    def test_vif(self):
        print 'test vif...'
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44417014583333336)
        self.assertAlmostEquals(scores[0], 0.574283)
        self.assertAlmostEquals(scores[1], 0.491295)

    def test_all(self):
        print 'test all...'
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44417014583333336)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666)

    def test_psnr(self):
        print 'test psnr...'
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=self.PSNR, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "psnr")
        self.assertAlmostEquals(score, 30.755063979166664)
        self.assertAlmostEquals(scores[0], 34.760779)
        self.assertAlmostEquals(scores[1], 31.883227)

    def test_2nd_moment(self):
        print 'test 2nd moment...'
        cmd = "{moment} 2 {fmt} {dis} {w} {h} > {log}".format(
            moment=self.MOMENT, fmt=self.YUV_FMT, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "1stmoment")
        self.assertAlmostEquals(score, 61.332006624999984)
        score, scores = read_log(self.LOG_FILENAME, "2ndmoment")
        self.assertAlmostEquals(score, 4798.659574041666)

class FeatureTestYuv422p10le(unittest.TestCase):

    VMAF = config.ROOT + "/feature/vmaf"
    PSNR = config.ROOT + "/feature/psnr"
    MOMENT = config.ROOT + "/feature/moment"
    LOG_FILENAME = config.ROOT + "/workspace/log"
    REF_YUV = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv422p10le.yuv"
    DIS_YUV = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv422p10le.yuv"
    YUV_FMT = "yuv422p10le"
    YUV_WIDTH = 576
    YUV_HEIGHT = 324

    def setUp(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def tearDown(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def test_adm(self):
        print 'test adm on yuv422p10le...'
        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666)

    def test_ansnr(self):
        print 'test ansnr on yuv422p10le...'
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333)

    def test_motion(self):
        print 'test motion on yuv422p10le...'
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667)

    def test_vif(self):
        print 'test vif on yuv422p10le...'
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44417014583333336)
        self.assertAlmostEquals(scores[0], 0.574283)
        self.assertAlmostEquals(scores[1], 0.491295)

    def test_all(self):
        print 'test all on yuv422p10le...'
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44417014583333336)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666)

    def test_psnr(self):
        print 'test psnr on yuv422p10le...'
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=self.PSNR, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "psnr")
        self.assertAlmostEquals(score, 30.78057329166666)
        self.assertAlmostEquals(scores[0], 34.786288)
        self.assertAlmostEquals(scores[1], 31.908737)

    def test_2nd_moment(self):
        print 'test 2nd moment on yuv422p10le...'
        cmd = "{moment} 2 {fmt} {dis} {w} {h} > {log}".format(
            moment=self.MOMENT, fmt=self.YUV_FMT, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "1stmoment")
        self.assertAlmostEquals(score, 61.332006624999984)
        score, scores = read_log(self.LOG_FILENAME, "2ndmoment")
        self.assertAlmostEquals(score, 4798.659574041666)


class CornerCaseTest(unittest.TestCase):

    VMAF = config.ROOT + "/feature/vmaf"
    LOG_FILENAME = config.ROOT + "/workspace/log"
    CMD_TEMPLATE = """
        {vmaf} vif {fmt} {ref} {dis} {w} {h} > {log};
        {vmaf} adm {fmt} {ref} {dis} {w} {h} >> {log};
        {vmaf} ansnr {fmt} {ref} {dis} {w} {h} >> {log};
        {vmaf} motion {fmt} {ref} {dis} {w} {h} >> {log};"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def test_checkerboard_identical(self):
        print 'test on checkerboard pattern identical...'
        ref_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 25.583514666666662)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0)

    def test_checkerboard_shifted_by_1(self):
        print 'test on checkerboard pattern shifted by 1...'
        ref_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 0.81386000000000003)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 12.418291000000002)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 0.15612933333333334)

    def test_checkerboard_opposite(self):
        print 'test on checkerboard pattern opposite...'
        ref_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_yuv = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_10_0.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 0.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], -1.2655523333333332)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 0.0)

    def test_flat_identical(self):
        print 'test on flat pattern identical...'
        ref_yuv = config.ROOT + "//resource/yuv/flat_1920_1080_0.yuv"
        dis_yuv = config.ROOT + "//resource/yuv/flat_1920_1080_0.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 49.967601999999999)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 0.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0)

    def test_flat_identical(self):
        print 'test on flat pattern identical...'
        ref_yuv = config.ROOT + "/resource/yuv/flat_1920_1080_0.yuv"
        dis_yuv = config.ROOT + "/resource/yuv/flat_1920_1080_0.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 49.967601999999999)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 0.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0)

    def test_flat_value10(self):
        print 'test on flat pattern of value 10...'
        ref_yuv = config.ROOT + "/resource/yuv/flat_1920_1080_0.yuv"
        dis_yuv = config.ROOT + "/resource/yuv/flat_1920_1080_10.yuv"
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=self.VMAF, fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=self.LOG_FILENAME)
        subprocess.call(cmd, shell=True)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 5.0022209999999997)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 0.0)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0)

if __name__ == '__main__':

    unittest.main()

    print 'Done.'
