from vmaf.tools.misc import run_process

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import re
import unittest

from vmaf.config import VmafConfig
from vmaf import ExternalProgram, required

REMOVE_LOG = 1 # for debug, make this 0

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
        print('test adm...')
        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ADM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ADM_LOG, "adm")
        self.assertAlmostEquals(score, 0.9345877708333336, places=4)
        score, scores = read_log(ADM_LOG, "adm_num")
        self.assertAlmostEquals(score, 371.8354140624999, places=4)
        score, scores = read_log(ADM_LOG, "adm_den")
        self.assertAlmostEquals(score, 397.8337897291667, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale0")
        self.assertAlmostEquals(score, 45.5277493125, places=4)

        score, scores = read_log(ADM_LOG, "adm_den_scale0")
        self.assertAlmostEquals(score, 50.143851375000004, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale1")
        self.assertAlmostEquals(score, 66.58064533333334, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale1")
        self.assertAlmostEquals(score, 74.47438285416666, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale2")
        self.assertAlmostEquals(score, 105.56477879166668, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale2")
        self.assertAlmostEquals(score, 113.49725852083333, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale3")
        self.assertAlmostEquals(score, 154.16224066666666, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale3")
        self.assertAlmostEquals(score, 159.71829710416668, places=4)

    def test_ansnr(self):
        ANSNR_LOG = self.LOG_FILENAME + '_ansnr'
        print('test ansnr...')
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ANSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ANSNR_LOG, "ansnr")
        self.assertAlmostEquals(score, 23.5095715208, places=4)
        score, scores = read_log(ANSNR_LOG, "anpsnr")
        self.assertAlmostEquals(score, 34.164776875, places=4)

    def test_motion(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion'
        print('test motion...')
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion")
        self.assertAlmostEquals(score, 4.04982535417, places=4)

    def test_motion2(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion2'
        print('test motion2...')
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion2")
        self.assertAlmostEquals(score, 3.8953518541666665, places=4)

    def test_vif(self):
        VIF_LOG = self.LOG_FILENAME + '_vif'
        print('test vif...')
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=VIF_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(VIF_LOG, "vif")
        self.assertAlmostEquals(score, 0.4460930625000001, places=4)
        self.assertAlmostEquals(scores[0], 0.580304, places=4)
        self.assertAlmostEquals(scores[1], 0.492477, places=4)
        score, scores = read_log(VIF_LOG, "vif_num")
        self.assertAlmostEquals(score, 712650.023478, places=0)
        score, scores = read_log(VIF_LOG, "vif_den")
        self.assertAlmostEquals(score, 1597314.95249, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale0")
        self.assertAlmostEquals(score, 468101.509766, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale1")
        self.assertAlmostEquals(score, 184971.572266, places=1)
        score, scores = read_log(VIF_LOG, "vif_num_scale2")
        self.assertAlmostEquals(score, 47588.8323567, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale3")
        self.assertAlmostEquals(score, 11988.1090902, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale0")
        self.assertAlmostEquals(score, 1287822.80208, places=0)
        score, scores = read_log(VIF_LOG, "vif_den_scale1")
        self.assertAlmostEquals(score, 241255.067708, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale2")
        self.assertAlmostEquals(score, 55149.8169759, places=2)
        score, scores = read_log(VIF_LOG, "vif_den_scale3")
        self.assertAlmostEquals(score, 13087.2657267, places=2)

    def test_all(self):
        ALL_LOG = self.LOG_FILENAME + "_all"
        print('test all...')
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ALL_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ALL_LOG, "vif")
        self.assertAlmostEquals(score, 0.4460930625, places=4)
        score, scores = read_log(ALL_LOG, "motion")
        self.assertAlmostEquals(score, 4.04982535417, places=4)
        score, scores = read_log(ALL_LOG, "motion2")
        self.assertAlmostEquals(score, 3.8953518541666665, places=4)
        score, scores = read_log(ALL_LOG, "ansnr")
        self.assertAlmostEquals(score, 23.509571520833337, places=4)
        score, scores = read_log(ALL_LOG, "adm")
        self.assertAlmostEquals(score, 0.9345877708333336, places=4)
        score, scores = read_log(ALL_LOG, "adm_num")
        self.assertAlmostEquals(score, 371.8354140624999, places=4)
        score, scores = read_log(ALL_LOG, "adm_den")
        self.assertAlmostEquals(score, 397.8337897291667, places=4)
        score, scores = read_log(ALL_LOG, "vif_num")
        self.assertAlmostEquals(score, 712650.023478, places=0)
        score, scores = read_log(ALL_LOG, "vif_den")
        self.assertAlmostEquals(score, 1597314.95249, places=0)
        score, scores = read_log(ALL_LOG, "anpsnr")
        self.assertAlmostEquals(score, 34.164776874999994, places=4)
        score, scores = read_log(ALL_LOG, "vif_num_scale0")
        self.assertAlmostEquals(score, 468101.509766, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale1")
        self.assertAlmostEquals(score, 184971.572266, places=1)
        score, scores = read_log(ALL_LOG, "vif_num_scale2")
        self.assertAlmostEquals(score, 47588.8323567, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale3")
        self.assertAlmostEquals(score, 11988.1090902, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale0")
        self.assertAlmostEquals(score, 1287822.80208, places=0)
        score, scores = read_log(ALL_LOG, "vif_den_scale1")
        self.assertAlmostEquals(score, 241255.067708, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale2")
        self.assertAlmostEquals(score, 55149.8169759, places=2)
        score, scores = read_log(ALL_LOG, "vif_den_scale3")
        self.assertAlmostEquals(score, 13087.2657267, places=2)
        score, scores = read_log(ALL_LOG, "adm_den_scale0")
        self.assertAlmostEquals(score, 50.143851375000004, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale1")
        self.assertAlmostEquals(score, 66.58064533333334, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale1")
        self.assertAlmostEquals(score, 74.47438285416666, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale2")
        self.assertAlmostEquals(score, 105.56477879166668, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale2")
        self.assertAlmostEquals(score, 113.49725852083333, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale3")
        self.assertAlmostEquals(score, 154.16224066666666, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale3")
        self.assertAlmostEquals(score, 159.71829710416668, places=4)

    def test_psnr(self):
        PSNR_LOG = self.LOG_FILENAME + '_psnr'
        print('test psnr...')
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=required(ExternalProgram.psnr), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=PSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(PSNR_LOG, "psnr")
        self.assertAlmostEquals(score, 30.7550639792, places=4)
        self.assertAlmostEquals(scores[0], 34.760779, places=4)
        self.assertAlmostEquals(scores[1], 31.88322, places=4)

    def test_2nd_moment(self):
        MOMENT_LOG = self.LOG_FILENAME + '_moment'
        print('test 2nd moment...')
        cmd = "{moment} 2 {fmt} {dis} {w} {h} > {log}".format(
            moment=required(ExternalProgram.moment), fmt=self.YUV_FMT, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOMENT_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOMENT_LOG, "1stmoment")
        self.assertAlmostEquals(score, 61.332006624999984, places=4)
        score, scores = read_log(MOMENT_LOG, "2ndmoment")
        self.assertAlmostEquals(score, 4798.659574041666, places=4)

    def test_ssim(self):
        SSIM_LOG = self.LOG_FILENAME + '_ssim'
        print('test ssim...')
        cmd = "{ssim} {fmt} {ref} {dis} {w} {h} > {log}".format(
            ssim=required(ExternalProgram.ssim), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=SSIM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(SSIM_LOG, "ssim")
        self.assertAlmostEquals(score, 0.863226541666667, places=4)
        self.assertAlmostEquals(scores[0], 0.925023, places=4)
        self.assertAlmostEquals(scores[1], 0.891992, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_l")
        self.assertAlmostEquals(score, 0.998147458333333, places=4)
        self.assertAlmostEquals(scores[0], 0.999524, places=4)
        self.assertAlmostEquals(scores[1], 0.998983, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_c")
        self.assertAlmostEquals(score, 0.9612679375000001, places=4)
        self.assertAlmostEquals(scores[0], 0.979614, places=4)
        self.assertAlmostEquals(scores[1], 0.96981, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_s")
        self.assertAlmostEquals(score, 0.8977363333333335, places=4)
        self.assertAlmostEquals(scores[0], 0.943966, places=4)
        self.assertAlmostEquals(scores[1], 0.919507, places=4)

    def test_ms_ssim(self):
        MS_SSIM_LOG = self.LOG_FILENAME + '_msssim'
        print('test ms_ssim...')
        cmd = "{ms_ssim} {fmt} {ref} {dis} {w} {h} > {log}".format(
            ms_ssim=required(ExternalProgram.ms_ssim), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MS_SSIM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim")
        self.assertAlmostEquals(score, 0.9632498125, places=4)
        self.assertAlmostEquals(scores[0], 0.981968, places=4)
        self.assertAlmostEquals(scores[1], 0.973366, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale0")
        self.assertAlmostEquals(score, 0.998147458333333, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale0")
        self.assertAlmostEquals(score, 0.9612679375000001, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale0")
        self.assertAlmostEquals(score, 0.8977363333333335, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale1")
        self.assertAlmostEquals(score, 0.9989961250000002, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale1")
        self.assertAlmostEquals(score, 0.9857694375, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale1")
        self.assertAlmostEquals(score, 0.941185875, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale2")
        self.assertAlmostEquals(score, 0.9992356458333332, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale2")
        self.assertAlmostEquals(score, 0.997034020833, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale2")
        self.assertAlmostEquals(score, 0.977992145833, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale3")
        self.assertAlmostEquals(score, 0.9992921041666665, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale3")
        self.assertAlmostEquals(score, 0.999588104167, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale3")
        self.assertAlmostEquals(score, 0.99387125, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale4")
        self.assertAlmostEquals(score, 0.9994035625000003, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale4")
        self.assertAlmostEquals(score, 0.999907625, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale4")
        self.assertAlmostEquals(score, 0.998222583333, places=4)

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
        print('test adm on yuv422p10le...')
        cmd = "{vmaf} adm {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ADM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ADM_LOG, "adm")
        self.assertAlmostEquals(score, 0.9345877708333336, places=4)
        score, scores = read_log(ADM_LOG, "adm_num")
        self.assertAlmostEquals(score, 371.8354140624999, places=4)
        score, scores = read_log(ADM_LOG, "adm_den")
        self.assertAlmostEquals(score, 397.8337897291667, places=4)

        score, scores = read_log(ADM_LOG, "adm_den_scale0")
        self.assertAlmostEquals(score, 50.143851375000004, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale1")
        self.assertAlmostEquals(score, 66.58064533333334, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale1")
        self.assertAlmostEquals(score, 74.47438285416666, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale2")
        self.assertAlmostEquals(score, 105.56477879166668, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale2")
        self.assertAlmostEquals(score, 113.49725852083333, places=4)
        score, scores = read_log(ADM_LOG, "adm_num_scale3")
        self.assertAlmostEquals(score, 154.16224066666666, places=4)
        score, scores = read_log(ADM_LOG, "adm_den_scale3")
        self.assertAlmostEquals(score, 159.71829710416668, places=4)

    def test_ansnr(self):
        ANSNR_LOG = self. LOG_FILENAME + '_ansnr'
        print('test ansnr on yuv422p10le...')
        cmd = "{vmaf} ansnr {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ANSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ANSNR_LOG, "ansnr")
        self.assertAlmostEquals(score, 23.5095715208, places=4)
        score, scores = read_log(ANSNR_LOG, "anpsnr")
        self.assertAlmostEquals(score, 34.1902860625, places=4)

    def test_motion(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion'
        print('test motion on yuv422p10le...')
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion")
        self.assertAlmostEquals(score, 4.04982535417, places=4)

    def test_motion2(self):
        MOTION_LOG = self.LOG_FILENAME + '_motion2'
        print('test motion2 on yuv422p10le...')
        cmd = "{vmaf} motion {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MOTION_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MOTION_LOG, "motion2")
        self.assertAlmostEquals(score, 3.8953518541666665, places=4)

    def test_vif(self):
        VIF_LOG = self.LOG_FILENAME + '_vif'
        print('test vif on yuv422p10le...')
        cmd = "{vmaf} vif {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=VIF_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(VIF_LOG, "vif")
        self.assertAlmostEquals(score, 0.4460930625, places=4)
        self.assertAlmostEquals(scores[0], 0.580304, places=4)
        self.assertAlmostEquals(scores[1], 0.492477, places=4)
        score, scores = read_log(VIF_LOG, "vif_num")
        self.assertAlmostEquals(score, 712650.023478, places=0)
        score, scores = read_log(VIF_LOG, "vif_den")
        self.assertAlmostEquals(score, 1597314.95249, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale0")
        self.assertAlmostEquals(score, 468101.509766, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale1")
        self.assertAlmostEquals(score, 184971.572266, places=1)
        score, scores = read_log(VIF_LOG, "vif_num_scale2")
        self.assertAlmostEquals(score, 47588.8323567, places=0)
        score, scores = read_log(VIF_LOG, "vif_num_scale3")
        self.assertAlmostEquals(score, 11988.1090902, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale0")
        self.assertAlmostEquals(score, 1287822.80208, places=0)
        score, scores = read_log(VIF_LOG, "vif_den_scale1")
        self.assertAlmostEquals(score, 241255.067708, places=1)
        score, scores = read_log(VIF_LOG, "vif_den_scale2")
        self.assertAlmostEquals(score, 55149.8169759, places=2)
        score, scores = read_log(VIF_LOG, "vif_den_scale3")
        self.assertAlmostEquals(score, 13087.2657267, places=2)

    def test_all(self):
        ALL_LOG = self.LOG_FILENAME + "_all"
        print('test all on yuv422p10le...')
        cmd = "{vmaf} all {fmt} {ref} {dis} {w} {h} > {log}".format(
            vmaf=required(ExternalProgram.vmaf), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=ALL_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(ALL_LOG, "vif")
        self.assertAlmostEquals(score, 0.4460930625, places=4)
        score, scores = read_log(ALL_LOG, "motion")
        self.assertAlmostEquals(score, 4.04982535417, places=4)
        score, scores = read_log(ALL_LOG, "motion2")
        self.assertAlmostEquals(score, 3.8953518541666665, places=4)
        score, scores = read_log(ALL_LOG, "ansnr")
        self.assertAlmostEquals(score, 23.5095715208, places=4)
        score, scores = read_log(ALL_LOG, "adm")
        self.assertAlmostEquals(score, 0.9345877708333336, places=4)
        score, scores = read_log(ALL_LOG, "adm_num")
        self.assertAlmostEquals(score, 371.8354140624999, places=4)
        score, scores = read_log(ALL_LOG, "adm_den")
        self.assertAlmostEquals(score, 397.8337897291667, places=4)
        score, scores = read_log(ALL_LOG, "vif_num")
        self.assertAlmostEquals(score, 712650.023478, places=0)
        score, scores = read_log(ALL_LOG, "vif_den")
        self.assertAlmostEquals(score, 1597314.95249, places=0)
        score, scores = read_log(ALL_LOG, "anpsnr")
        self.assertAlmostEquals(score, 34.1902860625, places=4)
        score, scores = read_log(ALL_LOG, "vif_num_scale0")
        self.assertAlmostEquals(score, 468101.509766, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale1")
        self.assertAlmostEquals(score, 184971.572266, places=1)
        score, scores = read_log(ALL_LOG, "vif_num_scale2")
        self.assertAlmostEquals(score, 47588.8323567, places=0)
        score, scores = read_log(ALL_LOG, "vif_num_scale3")
        self.assertAlmostEquals(score, 11988.1090902, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale0")
        self.assertAlmostEquals(score, 1287822.80208, places=0)
        score, scores = read_log(ALL_LOG, "vif_den_scale1")
        self.assertAlmostEquals(score, 241255.067708, places=1)
        score, scores = read_log(ALL_LOG, "vif_den_scale2")
        self.assertAlmostEquals(score, 55149.8169759, places=2)
        score, scores = read_log(ALL_LOG, "vif_den_scale3")
        self.assertAlmostEquals(score, 13087.2657267, places=2)
        score, scores = read_log(ALL_LOG, "adm_den_scale0")
        self.assertAlmostEquals(score, 50.143851375000004, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale1")
        self.assertAlmostEquals(score, 66.58064533333334, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale1")
        self.assertAlmostEquals(score, 74.47438285416666, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale2")
        self.assertAlmostEquals(score, 105.56477879166668, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale2")
        self.assertAlmostEquals(score, 113.49725852083333, places=4)
        score, scores = read_log(ALL_LOG, "adm_num_scale3")
        self.assertAlmostEquals(score, 154.16224066666666, places=4)
        score, scores = read_log(ALL_LOG, "adm_den_scale3")
        self.assertAlmostEquals(score, 159.71829710416668, places=4)

    def test_psnr(self):
        PSNR_LOG = self.LOG_FILENAME + '_psnr'
        print('test psnr on yuv422p10le...')
        cmd = "{psnr} {fmt} {ref} {dis} {w} {h} > {log}".format(
            psnr=required(ExternalProgram.psnr), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=PSNR_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(PSNR_LOG, "psnr")
        self.assertAlmostEquals(score, 30.7805732917, places=4)
        self.assertAlmostEquals(scores[0], 34.786288, places=4)
        self.assertAlmostEquals(scores[1], 31.908737, places=4)

    def test_ssim(self):
        SSIM_LOG = self.LOG_FILENAME + '_ssim'
        print('test ssim on yuv422p10le...')
        cmd = "{ssim} {fmt} {ref} {dis} {w} {h} > {log}".format(
            ssim=required(ExternalProgram.ssim), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=SSIM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(SSIM_LOG, "ssim")
        self.assertAlmostEquals(score, 0.863226541666667, places=4)
        self.assertAlmostEquals(scores[0], 0.925023, places=4)
        self.assertAlmostEquals(scores[1], 0.891992, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_l")
        self.assertAlmostEquals(score,0.998147458333333, places=4)
        self.assertAlmostEquals(scores[0], 0.999524, places=4)
        self.assertAlmostEquals(scores[1], 0.998983, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_c")
        self.assertAlmostEquals(score, 0.9612679375000001, places=4)
        self.assertAlmostEquals(scores[0], 0.979614, places=4)
        self.assertAlmostEquals(scores[1], 0.96981, places=4)
        score, scores = read_log(SSIM_LOG, "ssim_s")
        self.assertAlmostEquals(score, 0.8977363333333335, places=4)
        self.assertAlmostEquals(scores[0], 0.943966, places=4)
        self.assertAlmostEquals(scores[1], 0.919507, places=4)

    def test_ms_ssim(self):
        MS_SSIM_LOG = self.LOG_FILENAME + '_msssim'
        print('test ms_ssim on yuv422p10le...')
        cmd = "{ms_ssim} {fmt} {ref} {dis} {w} {h} > {log}".format(
            ms_ssim=required(ExternalProgram.ms_ssim), fmt=self.YUV_FMT, ref=self.REF_YUV, dis=self.DIS_YUV,
            w=self.YUV_WIDTH, h=self.YUV_HEIGHT, log=MS_SSIM_LOG
        )
        run_process(cmd, shell=True)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim")
        self.assertAlmostEquals(score, 0.9632498125, places=4)
        self.assertAlmostEquals(scores[0], 0.981968, places=4)
        self.assertAlmostEquals(scores[1], 0.973366, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale0")
        self.assertAlmostEquals(score, 0.998147458333333, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale0")
        self.assertAlmostEquals(score, 0.9612679375000001, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale0")
        self.assertAlmostEquals(score, 0.8977363333333335, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale1")
        self.assertAlmostEquals(score, 0.9989961250000002, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale1")
        self.assertAlmostEquals(score, 0.9857694375, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale1")
        self.assertAlmostEquals(score, 0.941185875, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale2")
        self.assertAlmostEquals(score, 0.9992356458333332, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale2")
        self.assertAlmostEquals(score, 0.997034020833, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale2")
        self.assertAlmostEquals(score, 0.977992145833, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale3")
        self.assertAlmostEquals(score, 0.9992921041666665, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale3")
        self.assertAlmostEquals(score, 0.9995884375000003, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale3")
        self.assertAlmostEquals(score, 0.9938712499999998, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_l_scale4")
        self.assertAlmostEquals(score, 0.9994035625000003, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_c_scale4")
        self.assertAlmostEquals(score, 0.999907625, places=4)
        score, scores = read_log(MS_SSIM_LOG, "ms_ssim_s_scale4")
        self.assertAlmostEquals(score,0.998222583333, places=4)


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
        print('test on checkerboard pattern identical...')
        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboardIdentical'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 21.1138813333, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.554836666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.554836666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 2773.891225, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 33021350.5, places=-3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 29.8567246667, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 25757432.0, places=-3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 259774.958333, places=1)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 277.120382, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 924.193766, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_checkerboard_shifted_by_1(self):
        print('test on checkerboard pattern shifted by 1...')
        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboard_shifted_by_1'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm")[0], 0.7853383333333334, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 7.92623066667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.5548366667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.5548366667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif")[0], 0.156834666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 2178.5352886666665, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 5178894.51562, places=-1)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 16.669074, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 2908829.0, places=-1)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 128957.796875, places=-2)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 201.15329999999997, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 765.1562903333333, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_checkerboard_opposite(self):
        print('test on checkerboard pattern opposite...')
        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_checkerboard_opposite'
        ref_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm")[0], 0.053996333333333334, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], -5.758091333333334, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion")[0], 12.554836666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 12.554836666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780313, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 2773.891225, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 6.66666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 33021387.0625, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 2.984752, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 6.66666666667, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 25757473.3333, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 259774.9375, places=3)

        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 277.120382, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667711, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 924.193766, places=3)

    def test_flat_identical(self):
        print('test on flat pattern identical...')
        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_flat_identical'
        ref_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 60.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780392, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 149.780392, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 2754000.15625, places=1)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 2754000.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 60.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0], 2073600.125, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 2073600.000, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 32400.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 32400.0, places=4)

        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 65.573967, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667747, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 16.667747, places=3)

    def test_flat_value10(self):
        print('test on flat pattern of value 10...')
        LOCAL_LOG_FILENAME = self.LOG_FILENAME + '_flat_value10'
        ref_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_yuv = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
        yuv_fmt = "yuv420p"
        yuv_width = 1920
        yuv_height = 1080
        cmd = self.CMD_TEMPLATE.format(vmaf=required(ExternalProgram.vmaf), fmt=yuv_fmt, ref=ref_yuv,
                                       dis=dis_yuv, w=yuv_width, h=yuv_height,
                                       log=LOCAL_LOG_FILENAME)
        run_process(cmd, shell=True)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "ansnr")[0], 21.899511, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "motion2")[0], 0.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif")[0], 1.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num")[0], 149.780313, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den")[0], 149.780392, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num")[0], 2753999.99219, places=1)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den")[0], 2754000.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "anpsnr")[0], 29.045954, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale0")[0],2073600.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale0")[0], 2073600.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_num_scale3")[0], 32400.0, places=4)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "vif_den_scale3")[0], 32400.0, places=4)

        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale0")[0], 65.573967, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale0")[0], 65.573967, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_num_scale3")[0], 16.667711, places=3)
        self.assertAlmostEquals(read_log(LOCAL_LOG_FILENAME, "adm_den_scale3")[0], 16.667747, places=3)

if __name__ == '__main__':

    unittest.main()

    print('Done.')
