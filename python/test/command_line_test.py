import os
import unittest
import subprocess

import config

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class CommandLineTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filename = config.ROOT + '/python/test/resource/example_dataset.py'
        self.raw_dataset_filename = config.ROOT + '/python/test/resource/example_raw_dataset.py'
        self.out_model_filepath = config.ROOT + '/workspace/workdir/tmp.pkl'
        self.param_filename = config.ROOT + '/python/test/resource/vmaf_v4.py'
        self.batch_filename = config.ROOT + '/workspace/workdir/test_batch_input'

    def tearDown(self):
        if os.path.exists(self.out_model_filepath):
            os.remove(self.out_model_filepath)
        if os.path.exists(self.out_model_filepath + '.model'):
            os.remove(self.out_model_filepath + '.model')
        if os.path.exists(self.batch_filename):
            os.remove(self.batch_filename)

    def test_run_testing_vmaf(self):
        exe = config.ROOT + '/run_testing'
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_testing_vmaf_raw_dataset(self):
        exe = config.ROOT + '/run_testing'
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.raw_dataset_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_testing_psnr(self):
        exe = config.ROOT + '/run_testing'
        cmd = "{exe} PSNR {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_training(self):
        exe = config.ROOT + '/run_vmaf_training'
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_training_raw_dataset(self):
        exe = config.ROOT + '/run_vmaf_training'
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.raw_dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_in_batch(self):
        line = 'yuv420p 576 324 {root}/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/resource/yuv/src01_hrc01_576x324.yuv'.format(root=config.ROOT)
        cmd = 'echo "{line}" > {batch_filename}'.format(
            line=line, batch_filename=self.batch_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

        exe = config.ROOT + '/run_vmaf_in_batch'
        cmd = "{exe} {input} --parallelize >/dev/null 2>&1".format(
            exe=exe, input=self.batch_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf(self):
        exe = config.ROOT + '/run_vmaf'
        line = 'yuv420p 576 324 {root}/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/resource/yuv/src01_hrc01_576x324.yuv'.format(root=config.ROOT)
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_psnr(self):
        exe = config.ROOT + '/run_psnr'
        line = 'yuv420p 576 324 {root}/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/resource/yuv/src01_hrc01_576x324.yuv'.format(root=config.ROOT)
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_cleaning_cache_psnr(self):
        exe = config.ROOT + '/run_testing'
        cmd = "{exe} PSNR {dataset} --parallelize --cache-result --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

        exe = config.ROOT + '/python/run_cleaning_cache.py'
        cmd = "{exe} PSNR {dataset}".format(
            exe=exe, dataset=self.dataset_filename)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 0)

if __name__ == '__main__':
    unittest.main()
