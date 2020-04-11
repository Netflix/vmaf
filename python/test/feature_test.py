from vmaf.tools.misc import run_process

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import os
import re
import unittest

from vmaf.config import VmafConfig
from vmaf import ExternalProgram, required

REMOVE_LOG = 1  # for debug, make this 0


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

    LOG_FILENAME = VmafConfig.workdir_path("logFeatureTest")
    REF_YUV = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    DIS_YUV = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    YUV_FMT = "yuv420p"
    YUV_WIDTH = 576
    YUV_HEIGHT = 324

    def setUp(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def tearDown(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

        if REMOVE_LOG:
            (logPath, logFilePrefix) = os.path.split(self.LOG_FILENAME)
            filenames = [filename for filename in os.listdir(logPath) if filename.startswith(logFilePrefix)]
            for filename in filenames:
                os.remove(os.path.join(logPath, filename))

    def test_adm(self):
        ADM_LOG = self.LOG_FILENAME + '_adm'

        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ADM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ADM_LOG, "adm")
        self.assertAlmostEqual(score, 0.9345877708333336, places=4)
        score, scores = read_log(ADM_LOG, "adm_num")
        self.assertAlmostEqual(score, 371.8354140624999, places=4)
        score, scores = read_log(ADM_LOG, "adm_den")
        self.assertAlmostEqual(score, 397.8337897291667, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale0")
        self.assertAlmostEqual(score, 45.5277493125, places=4)

        score, scores = read_log(ADM_LOG, "adm_den_scale0")
        self.assertAlmostEqual(score, 50.143851375000004, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale1")
        self.assertAlmostEqual(score, 66.58064533333334, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale1")
        self.assertAlmostEqual(score, 74.47438285416666, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale2")
        self.assertAlmostEqual(score, 105.56477879166668, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale2")
        self.assertAlmostEqual(score, 113.49725852083333, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale3")
        self.assertAlmostEqual(score, 154.16224066666666, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale3")
        self.assertAlmostEqual(score, 159.71829710416668, places=4)

    def test_ansnr(self):
        ANSNR_LOG = self.LOG_FILENAME + '_ansnr'

        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ANSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ANSNR_LOG, "ansnr")
        self.assertAlmostEqual(score, 23.5095715208, places=4)
        score, scores = read_log(ANSNR_LOG, "anpsnr")
        self.assertAlmostEqual(score, 34.164776875, places=4)

    def test_motion(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion'

        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion")
        self.assertAlmostEqual(score, 4.04982535417, places=4)

    def test_motion2(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion2'

        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion2")
        self.assertAlmostEqual(score, 3.8953518541666665, places=4)

    def test_vif(self):
        VIF_LOG = self.LOG_FILENAME + '_vif'

        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=VIF_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(VIF_LOG, "vif")
        self.assertAlmostEqual(score, 0.4460930625000001, places=4)
        self.assertAlmostEqual(scores[0], 0.580304, places=4)
        self.assertAlmostEqual(scores[1], 0.492477, places=4)
        score, scores = read_log(VIF_LOG, "vif_num")
        self.assertAlmostEqual(score, 712650.023478, places=0)
        score, scores = read_log(VIF_LOG, "vif_den")
        self.assertAlmostEqual(score, 1597314.95249, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale0")
        self.assertAlmostEqual(score, 468101.509766, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale1")
        self.assertAlmostEqual(score, 184971.572266, places=1)
        score, scores = read_log(VIF_LOG, "vif_num_scale2")
        self.assertAlmostEqual(score, 47588.8323567, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale3")
        self.assertAlmostEqual(score, 11988.1090902, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale0")
        self.assertAlmostEqual(score, 1287822.80208, places=0)
        score, scores = read_log(VIF_LOG, "vif_den_scale1")
        self.assertAlmostEqual(score, 241255.067708, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale2")
        self.assertAlmostEqual(score, 55149.8169759, places=2)
        score, scores = read_log(VIF_LOG, "vif_den_scale3")
        self.assertAlmostEqual(score, 13087.2657267, places=2)

    def test_all(self):
        ALL_LOG = self.LOG_FILENAME + "_all"

        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ALL_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ALL_LOG, "vif")
        self.assertAlmostEqual(score, 0.4460930625, places=4)
        score, scores = read_log(ALL_LOG, "motion")
        self.assertAlmostEqual(score, 4.04982535417, places=4)
        score, scores = read_log(ALL_LOG, "motion2")
        self.assertAlmostEqual(score, 3.8953518541666665, places=4)
        score, scores = read_log(ALL_LOG, "ansnr")
        self.assertAlmostEqual(score, 23.509571520833337, places=4)
        score, scores = read_log(ALL_LOG, "adm")
        self.assertAlmostEqual(score, 0.9345877708333336, places=4)
        score, scores = read_log(ALL_LOG, "adm_num")
        self.assertAlmostEqual(score, 371.8354140624999, places=4)
        score, scores = read_log(ALL_LOG, "adm_den")
        self.assertAlmostEqual(score, 397.8337897291667, places=4)
        score, scores = read_log(ALL_LOG, "vif_num")
        self.assertAlmostEqual(score, 712650.023478, places=0)
        score, scores = read_log(ALL_LOG, "vif_den")
        self.assertAlmostEqual(score, 1597314.95249, places=0)
        score, scores = read_log(ALL_LOG, "anpsnr")
        self.assertAlmostEqual(score, 34.164776874999994, places=4)
        score, scores = read_log(ALL_LOG, "vif_num_scale0")
        self.assertAlmostEqual(score, 468101.509766, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale1")
        self.assertAlmostEqual(score, 184971.572266, places=1)
        score, scores = read_log(ALL_LOG, "vif_num_scale2")
        self.assertAlmostEqual(score, 47588.8323567, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale3")
        self.assertAlmostEqual(score, 11988.1090902, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale0")
        self.assertAlmostEqual(score, 1287822.80208, places=0)
        score, scores = read_log(ALL_LOG, "vif_den_scale1")
        self.assertAlmostEqual(score, 241255.067708, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale2")
        self.assertAlmostEqual(score, 55149.8169759, places=2)
        score, scores = read_log(ALL_LOG, "vif_den_scale3")
        self.assertAlmostEqual(score, 13087.2657267, places=2)
        score, scores = read_log(ALL_LOG, "adm_den_scale0")
        self.assertAlmostEqual(score, 50.143851375000004, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale1")
        self.assertAlmostEqual(score, 66.58064533333334, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale1")
        self.assertAlmostEqual(score, 74.47438285416666, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale2")
        self.assertAlmostEqual(score, 105.56477879166668, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale2")
        self.assertAlmostEqual(score, 113.49725852083333, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale3")
        self.assertAlmostEqual(score, 154.16224066666666, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale3")
        self.assertAlmostEqual(score, 159.71829710416668, places=4)


class FeatureTestYuv422p10le(unittest.TestCase):

    LOG_FILENAME = VmafConfig.workdir_path("logFeatureTestYuv422p10le")
    REF_YUV = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv")
    DIS_YUV = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv")
    YUV_FMT = "yuv422p10le"
    YUV_WIDTH = 576
    YUV_HEIGHT = 324

    def setUp(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

    def tearDown(self):
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

        if (REMOVE_LOG):
            (logPath, logFilePrefix) = os.path.split(self.LOG_FILENAME)
            filenames = [filename for filename in os.listdir(logPath) if filename.startswith(logFilePrefix)]
            for filename in filenames:
                os.remove(os.path.join(logPath, filename))

    def test_adm(self):
        ADM_LOG = self.LOG_FILENAME + '_adm'

        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ADM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ADM_LOG, "adm")
        self.assertAlmostEqual(score, 0.9345877708333336, places=4)
        score, scores = read_log(ADM_LOG, "adm_num")
        self.assertAlmostEqual(score, 371.8354140624999, places=4)
        score, scores = read_log(ADM_LOG, "adm_den")
        self.assertAlmostEqual(score, 397.8337897291667, places=4)

        score, scores = read_log(ADM_LOG, "adm_den_scale0")
        self.assertAlmostEqual(score, 50.143851375000004, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale1")
        self.assertAlmostEqual(score, 66.58064533333334, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale1")
        self.assertAlmostEqual(score, 74.47438285416666, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale2")
        self.assertAlmostEqual(score, 105.56477879166668, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale2")
        self.assertAlmostEqual(score, 113.49725852083333, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale3")
        self.assertAlmostEqual(score, 154.16224066666666, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale3")
        self.assertAlmostEqual(score, 159.71829710416668, places=4)

    def test_ansnr(self):
        ANSNR_LOG = self. LOG_FILENAME + '_ansnr'

        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ANSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ANSNR_LOG, "ansnr")
        self.assertAlmostEqual(score, 23.5095715208, places=4)
        score, scores = read_log(ANSNR_LOG, "anpsnr")
        self.assertAlmostEqual(score, 34.1902860625, places=4)

    def test_motion(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion'

        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion")
        self.assertAlmostEqual(score, 4.04982535417, places=4)

    def test_motion2(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion2'

        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion2")
        self.assertAlmostEqual(score, 3.8953518541666665, places=4)

    def test_vif(self):
        VIF_LOG = self.LOG_FILENAME + '_vif'

        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=VIF_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(VIF_LOG, "vif")
        self.assertAlmostEqual(score, 0.4460930625, places=4)
        self.assertAlmostEqual(scores[0], 0.580304, places=4)
        self.assertAlmostEqual(scores[1], 0.492477, places=4)
        score, scores = read_log(VIF_LOG, "vif_num")
        self.assertAlmostEqual(score, 712650.023478, places=0)
        score, scores = read_log(VIF_LOG, "vif_den")
        self.assertAlmostEqual(score, 1597314.95249, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale0")
        self.assertAlmostEqual(score, 468101.509766, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale1")
        self.assertAlmostEqual(score, 184971.572266, places=1)
        score, scores = read_log(VIF_LOG, "vif_num_scale2")
        self.assertAlmostEqual(score, 47588.8323567, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale3")
        self.assertAlmostEqual(score, 11988.1090902, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale0")
        self.assertAlmostEqual(score, 1287822.80208, places=0)
        score, scores = read_log(VIF_LOG, "vif_den_scale1")
        self.assertAlmostEqual(score, 241255.067708, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale2")
        self.assertAlmostEqual(score, 55149.8169759, places=2)
        score, scores = read_log(VIF_LOG, "vif_den_scale3")
        self.assertAlmostEqual(score, 13087.2657267, places=2)

    def test_all(self):
        ALL_LOG = self.LOG_FILENAME + "_all"

        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf_feature), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ALL_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ALL_LOG, "vif")
        self.assertAlmostEqual(score, 0.4460930625, places=4)
        score, scores = read_log(ALL_LOG, "motion")
        self.assertAlmostEqual(score, 4.04982535417, places=4)
        score, scores = read_log(ALL_LOG, "motion2")
        self.assertAlmostEqual(score, 3.8953518541666665, places=4)
        score, scores = read_log(ALL_LOG, "ansnr")
        self.assertAlmostEqual(score, 23.5095715208, places=4)
        score, scores = read_log(ALL_LOG, "adm")
        self.assertAlmostEqual(score, 0.9345877708333336, places=4)
        score, scores = read_log(ALL_LOG, "adm_num")
        self.assertAlmostEqual(score, 371.8354140624999, places=4)
        score, scores = read_log(ALL_LOG, "adm_den")
        self.assertAlmostEqual(score, 397.8337897291667, places=4)
        score, scores = read_log(ALL_LOG, "vif_num")
        self.assertAlmostEqual(score, 712650.023478, places=0)
        score, scores = read_log(ALL_LOG, "vif_den")
        self.assertAlmostEqual(score, 1597314.95249, places=0)
        score, scores = read_log(ALL_LOG, "anpsnr")
        self.assertAlmostEqual(score, 34.1902860625, places=4)
        score, scores = read_log(ALL_LOG, "vif_num_scale0")
        self.assertAlmostEqual(score, 468101.509766, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale1")
        self.assertAlmostEqual(score, 184971.572266, places=1)
        score, scores = read_log(ALL_LOG, "vif_num_scale2")
        self.assertAlmostEqual(score, 47588.8323567, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale3")
        self.assertAlmostEqual(score, 11988.1090902, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale0")
        self.assertAlmostEqual(score, 1287822.80208, places=0)
        score, scores = read_log(ALL_LOG, "vif_den_scale1")
        self.assertAlmostEqual(score, 241255.067708, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale2")
        self.assertAlmostEqual(score, 55149.8169759, places=2)
        score, scores = read_log(ALL_LOG, "vif_den_scale3")
        self.assertAlmostEqual(score, 13087.2657267, places=2)
        score, scores = read_log(ALL_LOG, "adm_den_scale0")
        self.assertAlmostEqual(score, 50.143851375000004, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale1")
        self.assertAlmostEqual(score, 66.58064533333334, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale1")
        self.assertAlmostEqual(score, 74.47438285416666, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale2")
        self.assertAlmostEqual(score, 105.56477879166668, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale2")
        self.assertAlmostEqual(score, 113.49725852083333, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale3")
        self.assertAlmostEqual(score, 154.16224066666666, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale3")
        self.assertAlmostEqual(score, 159.71829710416668, places=4)


class CornerCaseTest(unittest.TestCase):

    LOG_FILENAME = VmafConfig.workdir_path("logCornerCaseTest")
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

        if (REMOVE_LOG):
            (logPath, logFilePrefix) = os.path.split(self.LOG_FILENAME)
            filenames = [filename for filename in os.listdir(logPath) if filename.startswith(logFilePrefix)]
            for filename in filenames:
                os.remove(os.path.join(logPath, filename))

    def test_checkerboard_identical(self):

        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboardIdentical'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf_feature), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 21.1138813333, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.554836666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.554836666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 2773.891225, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 33021350.5, places=-3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 29.8567246667, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 25757432.0, places=-3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 259774.958333, places=1)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 277.120382, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 924.193766, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_checkerboard_shifted_by_1(self):

        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboard_shifted_by_1'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf_feature), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm")[0], 0.7853383333333334, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 7.92623066667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.5548366667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.5548366667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif")[0], 0.156834666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 2178.5352886666665, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 5178894.51562, places=-1)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 16.669074, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 2908829.0, places=-1)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 128957.796875, places=-2)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 201.15329999999997, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 765.1562903333333, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_checkerboard_opposite(self):

        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboard_opposite'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf_feature), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm")[0], 0.053996333333333334, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], -5.758091333333334, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.554836666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.554836666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780313, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 6.66666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 2.984752, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 6.66666666667, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667711, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_flat_identical(self):

        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_flat_identical'
        ref_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf_feature), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 60.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780392, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 149.780392, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 2754000.15625, places=1)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 2754000.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 60.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 2073600.125, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 2073600.000, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 32400.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 32400.0, places=4)

        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 65.573967, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667747, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 16.667747, places=3)

    def test_flat_value10(self):

        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_flat_value10'
        ref_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf_feature), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 21.899511, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 0.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780313, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 149.780392, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 2753999.99219, places=1)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 2754000.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 29.045954, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0],2073600.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 2073600.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 32400.0, places=4)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 32400.0, places=4)

        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 65.573967, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667711, places=3)
        self.assertAlmostEqual(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 16.667747, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
