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
        self.assertAlmostEquals(score, 0.9155242291666666, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_num")
        self.assertAlmostEquals(score, 6899.815530270836, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_den")
        self.assertAlmostEquals(score, 7535.801140312499, places=4)

    def test_ansnr(self):
        print 'test ansnr...'
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333, places=4)
        score, scores = read_log(self.LOG_FILENAME, "anpsnr")
        self.assertAlmostEquals(score, 34.15266368750002, places=4)

    def test_motion(self):
        print 'test motion...'
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667, places=4)

    def test_vif(self):
        print 'test vif...'
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44455808333333313, places=4)
        self.assertAlmostEquals(scores[0], 0.574563, places=4)
        self.assertAlmostEquals(scores[1], 0.491594, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num")
        self.assertAlmostEquals(score, 644527.3311971038, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den")
        self.assertAlmostEquals(score, 1449635.3812459996, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale0")
        self.assertAlmostEquals(score, 432433.361328125, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale1")
        self.assertAlmostEquals(score, 162845.15657552084, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale2")
        self.assertAlmostEquals(score, 39775.62239575001, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale3")
        self.assertAlmostEquals(score, 9473.190897687497, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale0")
        self.assertAlmostEquals(score, 1182666.5, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale1")
        self.assertAlmostEquals(score, 210865.31412760416, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale2")
        self.assertAlmostEquals(score, 45814.93636072915, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale3")
        self.assertAlmostEquals(score, 10288.630757645837, places=4)

    def test_all(self):
        print 'test all...'
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44455808333333313, places=4)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667, places=4)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_num")
        self.assertAlmostEquals(score, 6899.815530270836, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_den")
        self.assertAlmostEquals(score, 7535.801140312499, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num")
        self.assertAlmostEquals(score, 644527.3311971038, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den")
        self.assertAlmostEquals(score, 1449635.3812459996, places=4)
        score, scores = read_log(self.LOG_FILENAME, "anpsnr")
        self.assertAlmostEquals(score, 34.15266368750002, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale0")
        self.assertAlmostEquals(score, 432433.361328125, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale1")
        self.assertAlmostEquals(score, 162845.15657552084, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale2")
        self.assertAlmostEquals(score, 39775.62239575001, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale3")
        self.assertAlmostEquals(score, 9473.190897687497, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale0")
        self.assertAlmostEquals(score, 1182666.5, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale1")
        self.assertAlmostEquals(score, 210865.31412760416, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale2")
        self.assertAlmostEquals(score, 45814.93636072915, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale3")
        self.assertAlmostEquals(score, 10288.630757645837, places=4)

    def test_psnr(self):
        print 'test psnr...'
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=self.PSNR, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "psnr")
        self.assertAlmostEquals(score, 30.755063979166664, places=4)
        self.assertAlmostEquals(scores[0], 34.760779, places=4)
        self.assertAlmostEquals(scores[1], 31.883227, places=4)

    def test_2nd_moment(self):
        print 'test 2nd moment...'
        cmd = "{moment} 2 {fmt} {dis} {w} {h} > {log}".format(
            moment=self.MOMENT, fmt=self.YUV_FMT, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "1stmoment")
        self.assertAlmostEquals(score, 61.332006624999984, places=4)
        score, scores = read_log(self.LOG_FILENAME, "2ndmoment")
        self.assertAlmostEquals(score, 4798.659574041666, places=4)

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
        self.assertAlmostEquals(score, 0.9155242291666666, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_num")
        self.assertAlmostEquals(score, 6899.815530270836, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_den")
        self.assertAlmostEquals(score, 7535.801140312499, places=4)

    def test_ansnr(self):
        print 'test ansnr on yuv422p10le...'
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333, places=4)
        score, scores = read_log(self.LOG_FILENAME, "anpsnr")
        self.assertAlmostEquals(score, 34.17817281250001, places=4)

    def test_motion(self):
        print 'test motion on yuv422p10le...'
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667, places=4)

    def test_vif(self):
        print 'test vif on yuv422p10le...'
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44455808333333313, places=4)
        self.assertAlmostEquals(scores[0], 0.574563, places=4)
        self.assertAlmostEquals(scores[1], 0.491594, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num")
        self.assertAlmostEquals(score, 644527.3311971038, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den")
        self.assertAlmostEquals(score, 1449635.3812459996, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale0")
        self.assertAlmostEquals(score, 432433.361328125, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale1")
        self.assertAlmostEquals(score, 162845.15657552084, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale2")
        self.assertAlmostEquals(score, 39775.62239575001, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale3")
        self.assertAlmostEquals(score, 9473.190897687497, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale0")
        self.assertAlmostEquals(score, 1182666.5, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale1")
        self.assertAlmostEquals(score, 210865.31412760416, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale2")
        self.assertAlmostEquals(score, 45814.93636072915, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale3")
        self.assertAlmostEquals(score, 10288.630757645837, places=4)

    def test_all(self):
        print 'test all on yuv422p10le...'
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=self.VMAF, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "vif")
        self.assertAlmostEquals(score, 0.44455808333333313, places=4)
        score, scores = read_log(self.LOG_FILENAME, "motion")
        self.assertAlmostEquals(score, 3.5916076041666667, places=4)
        score, scores = read_log(self.LOG_FILENAME, "ansnr")
        self.assertAlmostEquals(score, 22.53345677083333, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm")
        self.assertAlmostEquals(score, 0.9155242291666666, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_num")
        self.assertAlmostEquals(score, 6899.815530270836, places=4)
        score, scores = read_log(self.LOG_FILENAME, "adm_den")
        self.assertAlmostEquals(score, 7535.801140312499, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num")
        self.assertAlmostEquals(score, 644527.3311971038, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den")
        self.assertAlmostEquals(score, 1449635.3812459996, places=4)
        score, scores = read_log(self.LOG_FILENAME, "anpsnr")
        self.assertAlmostEquals(score, 34.17817281250001, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale0")
        self.assertAlmostEquals(score, 432433.361328125, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale1")
        self.assertAlmostEquals(score, 162845.15657552084, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale2")
        self.assertAlmostEquals(score, 39775.62239575001, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_num_scale3")
        self.assertAlmostEquals(score, 9473.190897687497, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale0")
        self.assertAlmostEquals(score, 1182666.5, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale1")
        self.assertAlmostEquals(score, 210865.31412760416, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale2")
        self.assertAlmostEquals(score, 45814.93636072915, places=4)
        score, scores = read_log(self.LOG_FILENAME, "vif_den_scale3")
        self.assertAlmostEquals(score, 10288.630757645837, places=4)

    def test_psnr(self):
        print 'test psnr on yuv422p10le...'
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=self.PSNR, fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "psnr")
        self.assertAlmostEquals(score, 30.78057329166666, places=4)
        self.assertAlmostEquals(scores[0], 34.786288, places=4)
        self.assertAlmostEquals(scores[1], 31.908737, places=4)

    def test_2nd_moment(self):
        print 'test 2nd moment on yuv422p10le...'
        cmd = "{moment} 2 {fmt} {dis} {w} {h} > {log}".format(
            moment=self.MOMENT, fmt=self.YUV_FMT, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=self.LOG_FILENAME
        )
        subprocess.call(cmd, shell=True)
        score, scores = read_log(self.LOG_FILENAME, "1stmoment")
        self.assertAlmostEquals(score, 61.332006624999984, places=4)
        score, scores = read_log(self.LOG_FILENAME, "2ndmoment")
        self.assertAlmostEquals(score, 4798.659574041666, places=4)


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
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 25.583514666666662, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_num")[0], 30814.90966033333, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_den")[0], 30814.90966033333, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num")[0], 32164040.489583332, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den")[0], 32164035.723958332, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "anpsnr")[0], 29.840877000000003, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale0")[0], 25175559.333333332, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale0")[0], 25175551.333333332, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale3")[0], 243577.94791666666, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale3")[0], 243578.015625, places=4)

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
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 0.81386000000000003, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 12.418291000000002, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 0.15612933333333334, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_num")[0], 25079.63600833334, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_den")[0], 30814.90966033333, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num")[0], 5021740.846354, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den")[0], 32164035.723958332, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "anpsnr")[0], 16.675653999999998, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale0")[0], 2838609.75, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale0")[0], 25175551.333333332, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale3")[0], 121630.71093733334, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale3")[0], 243578.015625, places=3)

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
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], -1.2655523333333332, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 12.343795333333333, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_num")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_den")[0], 30814.90966033333, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den")[0], 32164035.723958332, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "anpsnr")[0], 2.9918100000000005, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale0")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale0")[0], 25175551.333333332, places=3)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale3")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale3")[0], 243578.015625, places=3)

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
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 49.967601999999999, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_num")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_den")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num")[0], 1694463.882812, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den")[0], 1694463.882812, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "anpsnr")[0], 60.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale0")[0], 1280578.25, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale0")[0], 1280578.25, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale3")[0], 19311.265625, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale3")[0], 19311.265625, places=4)

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
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "ansnr")[0], 5.0022209999999997, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_num")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "adm_den")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num")[0], 1694463.882812, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den")[0], 1694463.882812, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "anpsnr")[0], 29.056124, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale0")[0], 1280578.25, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale0")[0], 1280578.25, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_num_scale3")[0], 19311.265625, places=4)
        self.assertAlmostEquals(read_log(self.LOG_FILENAME, "vif_den_scale3")[0], 19311.265625, places=4)

if __name__ == '__main__':

    unittest.main()

    print 'Done.'
