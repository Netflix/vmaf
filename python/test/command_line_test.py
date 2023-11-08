import os
import tempfile
import unittest
import subprocess

from vmaf.config import VmafConfig
from vmaf.tools.misc import run_process
from vmaf import ExternalProgram

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class RunProcessTest(unittest.TestCase):

    def test_run_process(self):
        ret = run_process('echo hello', shell=True)
        self.assertEqual(ret, 0)

    def test_run_process_false_cmd(self):
        with self.assertRaises(AssertionError) as e:
            run_process('echoo hello', shell=True)
        self.assertTrue('Process returned 127, cmd: echoo hello' in e.exception.args[0])
        self.assertTrue('not found' in e.exception.args[0])


class CommandLineTest(unittest.TestCase):

    def setUp(self):
        self.dataset_filename = VmafConfig.test_resource_path('example_dataset.py')
        self.raw_dataset_filename = VmafConfig.test_resource_path('example_raw_dataset.py')
        self.out_model_filepath = VmafConfig.workdir_path('tmp.json')
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
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_testing_vmaf_raw_dataset(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} VMAF {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.raw_dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_testing_psnr(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} PSNR {dataset} --parallelize --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_testing_proccesses0(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} PSNR {dataset} --parallelize --suppress-plot --processes 0".format(
            exe=exe, dataset=self.dataset_filename)
        with self.assertRaises(AssertionError):
            run_process(cmd, shell=True)

    def test_run_testing_proccesses2_without_parallelize(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} PSNR {dataset} --suppress-plot --processes 2".format(
            exe=exe, dataset=self.dataset_filename)
        with self.assertRaises(AssertionError):
            run_process(cmd, shell=True)

    def test_run_vmaf_training(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf_training.py')
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_vmaf_training_processes0(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf_training.py')
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot --processes 0".format(
            exe=exe,
            dataset=self.dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        with self.assertRaises(AssertionError):
            run_process(cmd, shell=True)

    def test_run_vmaf_training_processes2_without_parallelize(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf_training.py')
        cmd = "{exe} {dataset} {param} {param} {output} --suppress-plot --processes 2".format(
            exe=exe,
            dataset=self.dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        with self.assertRaises(AssertionError):
            run_process(cmd, shell=True)

    def test_run_vmaf_training_raw_dataset(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf_training.py')
        cmd = "{exe} {dataset} {param} {param} {output} --parallelize --suppress-plot".format(
            exe=exe,
            dataset=self.raw_dataset_filename,
            param=self.param_filename,
            output=self.out_model_filepath)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_vmaf(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf.py')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_vmaf_ci(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf.py')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_vmaf_both_local_explain_and_ci(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_vmaf.py')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} --local-explain --ci >/dev/null 2>&1".format(line=line, exe=exe)
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, 2)

    def test_run_psnr(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_psnr.py')
        line = 'yuv420p 576 324 {root}/python/test/resource/yuv/src01_hrc00_576x324.yuv ' \
               '{root}/python/test/resource/yuv/src01_hrc01_576x324.yuv'.format(root=VmafConfig.root_path())
        cmd = "{exe} {line} >/dev/null 2>&1".format(line=line, exe=exe)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

    def test_run_cleaning_cache_psnr(self):
        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_testing.py')
        cmd = "{exe} PSNR {dataset} --parallelize --cache-result --suppress-plot".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)

        exe = VmafConfig.root_path('python', 'vmaf', 'script', 'run_cleaning_cache.py')
        cmd = "{exe} PSNR {dataset}".format(
            exe=exe, dataset=self.dataset_filename)
        ret = run_process(cmd, shell=True)
        self.assertEqual(ret, 0)


class VmafexecCommandLineTest(unittest.TestCase):

    RC_SUCCESS = 0

    def setUp(self) -> None:
        self.output_file_path = tempfile.NamedTemporaryFile().name

    def tearDown(self) -> None:
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)

    def test_run_vmafexec(self):
        exe = ExternalProgram.vmafexec
        cmd = "{exe} --reference {ref} --distorted {dis} --width 576 --height 324 --pixel_format 420 --bitdepth 8 --xml --feature psnr " \
              "--model path={model} --quiet --output {output}".format(
            exe=exe,
            ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
            model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.json"),
            output=self.output_file_path)
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, self.RC_SUCCESS)
        with open(self.output_file_path, 'rt') as fo:
            fc = fo.read()
            self.assertTrue('<metric name="psnr_y" min="29.640688" max="34.760779" mean="30.755064" harmonic_mean="30.727905" />' in fc)

    def test_run_vmafexec_with_frame_skipping(self):
        exe = ExternalProgram.vmafexec
        cmd = "{exe} --reference {ref} --distorted {dis} --width 576 --height 324 --pixel_format 420 --bitdepth 8 --xml --feature psnr " \
              "--model path={model} --quiet --output {output} --frame_skip_ref 2 --frame_skip_dist 2".format(
            exe=exe,
            ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
            model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.json"),
            output=self.output_file_path)
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, self.RC_SUCCESS)
        with open(self.output_file_path, 'rt') as fo:
            fc = fo.read()
            self.assertTrue('<metric name="psnr_y" min="29.640688" max="33.788213" mean="30.643458" harmonic_mean="30.626214" />' in fc)

    def test_run_vmafexec_with_frame_skipping_unequal(self):
        exe = ExternalProgram.vmafexec
        cmd = "{exe} --reference {ref} --distorted {dis} --width 576 --height 324 --pixel_format 420 --bitdepth 8 --xml --feature psnr " \
              "--model path={model} --quiet --output {output} --frame_skip_ref 2 --frame_skip_dist 5".format(
            exe=exe,
            ref=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            dis=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
            model=VmafConfig.model_path("other_models", "vmaf_v0.6.0.json"),
            output=self.output_file_path)
        ret = subprocess.call(cmd, shell=True)
        self.assertEqual(ret, self.RC_SUCCESS)
        with open(self.output_file_path, 'rt') as fo:
            fc = fo.read()
            self.assertTrue('<metric name="psnr_y" min="19.019327" max="21.084954" mean="20.269606" harmonic_mean="20.258113" />' in fc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
