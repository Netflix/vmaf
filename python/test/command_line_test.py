import os
import unittest
import subprocess

from vmaf.config import VmafConfig
from vmaf.tools.misc import run_process

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class CommandLineTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filename = VmafConfig.test_resource_path('example_dataset.py')
        self.raw_dataset_filename = VmafConfig.test_resource_path('example_raw_dataset.py')
        self.out_model_filepath = VmafConfig.workdir_path('tmp.pkl')
        self.param_filename = VmafConfig.test_resource_path('vmaf_v4.py')
        self.batch_filename = VmafConfig.workdir_path('test_batch_input')

    def tearDown(self):
        if os.path.exists(self.out_model_filepath):
            os.remove(self.out_model_filepath)
        if os.path.exists(self.out_model_filepath + '.model'):
            os.remove(self.out_model_filepath + '.model')
        if os.path.exists(self.batch_filename):
            os.remove(self.batch_filename)

    def test_run_testing_vmaf(self):
        exe = VmafConfig.root_path('run_testing')
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_testing_vmaf_raw_dataset(self):
        exe = VmafConfig.root_path('run_testing')
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.raw_dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_testing_psnr(self):
        exe = VmafConfig.root_path('run_testing')
        cmd = "{exe} PSNR {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_training(self):
        exe = VmafConfig.root_path('run_vmaf_training')
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_training_raw_dataset(self):
        exe = VmafConfig.root_path('run_vmaf_training')
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.raw_dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_in_batch(self):
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = 'echo "{line}" > {batch_filename}'.format(
            line=line, batch_filename=self.batch_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

        exe = VmafConfig.root_path('run_vmaf_in_batch')
        cmd = "{exe} {input} --parallelize >/dev/null 2>&1".format(
            exe=exe, input=self.batch_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_in_batch_ci(self):
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = 'echo "{line}" > {batch_filename}'.format(
            line=line, batch_filename=self.batch_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

        exe = VmafConfig.root_path('run_vmaf_in_batch')
        cmd = "{exe} {input} --parallelize --ci >/dev/null 2>&1".format(
            exe=exe, input=self.batch_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf(self):
        exe = VmafConfig.root_path('run_vmaf')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_ci(self):
        exe = VmafConfig.root_path('run_vmaf')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_vmaf_both_local_explain_and_ci(self):
        exe = VmafConfig.root_path('run_vmaf')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --local-explain --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, 2)

    def test_run_psnr(self):
        exe = VmafConfig.root_path('run_psnr')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

    def test_run_cleaning_cache_psnr(self):
        exe = VmafConfig.root_path('run_testing')
        cmd = "{exe} PSNR {dataset} --parallelize --cache-result --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

        exe = VmafConfig.root_path('python', 'script', 'run_cleaning_cache.py')
        cmd = "{exe} PSNR {dataset}".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEquals(ret, 0)

class VmafossexecCommandLineTest(unittest.TestCase):

    RC_SUCCESS = 0
    RC_VMAF_EXCEPTION = 256 - 2
    RC_RUNTIME_ERROR = 256 - 3
    RC_LOGIC_ERROR = 256 - 4
    RC_SEGMENTATION_FAULT = 139
    RC_ARGUMENT_ISSUE = 1
    RC_MORE_ARGUMENT_ISSUE = 256 - 1

    def test_run_vmafossexec(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model} --thread 1 --subsample 2".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_SUCCESS)

    def test_run_vmafossexec_nonexist_model_file(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl_XXX"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_VMAF_EXCEPTION)

    def test_run_vmafossexec_wrong_model_fmt(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl.model"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_LOGIC_ERROR)

    def test_run_vmafossexec_nonexist_ref_file(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_XXX.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_ARGUMENT_ISSUE)

    def test_run_vmafossexec_nonexist_dis_file(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_XXX.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_ARGUMENT_ISSUE)

    def test_run_vmafossexec_unknown_yuv_fmt(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p_XXX 576 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_ARGUMENT_ISSUE)

    def test_run_vmafossexec_odd_resolution(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 575 323 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_ARGUMENT_ISSUE)

    def test_run_vmafossexec_wrong_wh_format(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 abc {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

    def test_run_vmafossexec_wrong_width_value(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 0 324 {ref} {dis} {model}".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

    def test_run_vmafossexec_unknown_log_fmt(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model} --log-fmt xml_XXX".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

    def test_run_vmafossexec_unknown_pooling(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model} --pool mean_XXX".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

    def test_run_vmafossexec_wrong_thread(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model} --thread -1 --subsample 2".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

    def test_run_vmafossexec_wrong_subsample(self):
        exe = VmafConfig.root_path('wrapper', 'vmafossexec')
        cmd = "{exe} yuv420p 576 324 {ref} {dis} {model} --thread 0 --subsample 0".format(
            exe=exe, ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"), model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl"))
        ret = subprocess.call(cmd, shell=True)
        self.assertEquals(ret, self.RC_MORE_ARGUMENT_ISSUE)

if __name__ == '__main__':
    unittest.main()
