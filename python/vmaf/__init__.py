import os
import subprocess
import logging

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"
__version__ = "2.0.0"

logging.basicConfig()
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
logger.setLevel("INFO")

try:
    from matplotlib import pyplot as plt
except BaseException:
    # TODO: importing matplotlib fails on OSX with system python, check what can be done there...
    # Error reported is:
    #   RuntimeError: Python is not installed as a framework.
    #   The Mac OS X backend will not be able to function correctly if Python is not installed as a framework.
    #   See the Python documentation for more information on installing Python as a framework on Mac OS X.
    #   Please either reinstall Python as a framework, or try one of the other backends.
    #   If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'.
    #   See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
    plt = None

from . import config

# Path to folder containing this file
VMAF_PYTHON_ROOT = os.path.dirname(os.path.abspath(__file__))


# Assuming vmaf source checkout, path to top checked out folder
VMAF_ROOT = os.path.abspath(os.path.join(VMAF_PYTHON_ROOT, '..', '..', ))


class ProcessRunner(object):

    def run(self, cmd, kwargs):
        try:
            logger.info(cmd)
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, **kwargs)
        except subprocess.CalledProcessError as e:
            raise AssertionError(f'Process returned {e.returncode}, cmd: {cmd}, kwargs: {kwargs}, msg: {str(e.output)}')


def run_process(cmd, **kwargs):
    process_runner = ProcessRunner()
    process_runner.run(cmd, kwargs)
    return 0


def project_path(relative_path):
    path = os.path.join(VMAF_ROOT, relative_path)
    return path


def required(path):
    if not os.path.exists(path):
        raise AssertionError("%s does not exist, did you build?" % (path))
    return path


def convert_pixel_format_ffmpeg2vmafexec(ffmpeg_pix_fmt):
    '''
    Convert FFmpeg-style pixel format (pix_fmt) to vmaf style.

    :param ffmpeg_pix_fmt: FFmpeg-style pixel format, for example: yuv420p, yuv420p10le
    :return: (pixel_format: str, bitdepth: int), for example: (420, 8), (420, 10)
    '''
    assert ffmpeg_pix_fmt in ['yuv420p', 'yuv422p', 'yuv444p',
                              'yuv420p10le', 'yuv422p10le', 'yuv444p10le',
                              'yuv420p12le', 'yuv422p12le', 'yuv444p12le',
                              'yuv420p16le', 'yuv422p16le', 'yuv444p16le',
                              ]

    if ffmpeg_pix_fmt in ['yuv420p', 'yuv420p10le', 'yuv420p12le', 'yuv420p16le']:
        pixel_format = '420'
    elif ffmpeg_pix_fmt in ['yuv422p', 'yuv422p10le', 'yuv422p12le', 'yuv422p16le']:
        pixel_format = '422'
    elif ffmpeg_pix_fmt in ['yuv444p', 'yuv444p10le', 'yuv444p12le', 'yuv444p16le']:
        pixel_format = '444'
    else:
        assert False

    if ffmpeg_pix_fmt in ['yuv420p', 'yuv422p', 'yuv444p']:
        bitdepth = 8
    elif ffmpeg_pix_fmt in ['yuv420p10le', 'yuv422p10le', 'yuv444p10le']:
        bitdepth = 10
    elif ffmpeg_pix_fmt in ['yuv420p12le', 'yuv422p12le', 'yuv444p12le']:
        bitdepth = 12
    elif ffmpeg_pix_fmt in ['yuv420p16le', 'yuv422p16le', 'yuv444p16le']:
        bitdepth = 16
    else:
        assert False
    return pixel_format, bitdepth


class ExternalProgram(object):
    """
    External C programs relied upon by the python vmaf code
    These external programs should be compiled before vmaf is ran, as per instructions in README
    """

    try:
        from . import externals
        external_vmaf_feature = config.VmafExternalConfig.vmaf_path()
        external_vmafexec = config.VmafExternalConfig.vmafexec_path()
    except ImportError:
        external_vmaf_feature = None
        external_vmafexec = None

    vmaf_feature = project_path(os.path.join("libvmaf", "build", "tools", "vmaf_feature")) if external_vmaf_feature is None else external_vmaf_feature
    vmafexec = project_path(os.path.join("libvmaf", "build", "tools", "vmaf")) if external_vmafexec is None else external_vmafexec


class ExternalProgramCaller(object):
    """
    Caller of ExternalProgram.
    """

    @staticmethod
    def call_vmafexec_single_feature(feature, yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None, options=None):
        options2 = {feature: options.copy() if options is not None else None}
        if options2[feature] is not None and 'disable_avx' in options2[feature]:
            options2['disable_avx'] = options2[feature]['disable_avx']
            del options2[feature]['disable_avx']
        if options2[feature] is not None and 'n_threads' in options2[feature]:
            options2['n_threads'] = options2[feature]['n_threads']
            del options2[feature]['n_threads']
        if options2[feature] is not None and '_open_workfile_method' in options2[feature]:
            options2['_open_workfile_method'] = options2[feature]['_open_workfile_method']
            del options2[feature]['_open_workfile_method']
        if options2[feature] is not None and '_close_workfile_method' in options2[feature]:
            options2['_close_workfile_method'] = options2[feature]['_close_workfile_method']
            del options2[feature]['_close_workfile_method']
        return ExternalProgramCaller.call_vmafexec_multi_features(
            [feature], yuv_type, ref_path, dis_path, w, h, log_file_path, logger=logger, options=options2)

    @staticmethod
    def call_vmafexec_multi_features(features, yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None, options=None):

        # ./libvmaf/build/tools/vmaf
        # --reference python/test/resource/yuv/src01_hrc00_576x324.yuv
        # --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv
        # --width 576 --height 324 --pixel_format 420 --bitdepth 8
        # --output /dev/stdout --xml --no_prediction --feature float_motion --feature integer_motion

        pixel_format, bitdepth = convert_pixel_format_ffmpeg2vmafexec(yuv_type)

        cmd = [
            required(ExternalProgram.vmafexec),
            '--reference', ref_path,
            '--distorted', dis_path,
            '--width', str(w),
            '--height', str(h),
            '--pixel_format', pixel_format,
            '--bitdepth', str(bitdepth),
            '--output', log_file_path,
            '--xml',
            '--no_prediction',
        ]

        if options is not None and 'disable_avx' in options:
            assert isinstance(options['disable_avx'], bool)
            if options['disable_avx'] is True:
                cmd += ['--cpumask', '-1']

        if options is not None and 'n_threads' in options:
            assert isinstance(options['n_threads'], int) and options['n_threads'] >= 1
            cmd += ['--threads', str(options['n_threads'])]

        for feature in features:
            if options is None:
                feature_str = feature
            else:
                assert isinstance(options, dict)
                if feature in options and options[feature] is not None and len(options[feature]) > 0:
                    assert isinstance(options[feature], dict)
                    options_lst = []
                    for k, v in options[feature].items():
                        if isinstance(v, bool):
                            v = str(v).lower()
                        options_lst.append(f'{k}={v}')
                    options_str = ':'.join(options_lst)
                    feature_str = '='.join([feature, options_str])
                else:
                    feature_str = feature
            cmd += ['--feature', feature_str]

        if logger:
            logger.info(' '.join(cmd))
        run_process(' '.join(cmd), shell=True)

    @staticmethod
    def call_vifdiff_feature(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        vifdiff_feature_cmd = "{vmaf} vifdiff {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            vmaf=required(ExternalProgram.vmaf_feature),
            yuv_type=yuv_type,
            ref_path=ref_path,
            dis_path=dis_path,
            w=w,
            h=h,
            log_file_path=log_file_path,
        )
        if logger:
            logger.info(vifdiff_feature_cmd)
        run_process(vifdiff_feature_cmd, shell=True)

    @staticmethod
    def call_vmafexec(reference, distorted, width, height, pixel_format, bitdepth,
                    float_psnr, psnr, float_ssim, ssim, float_ms_ssim, ms_ssim, float_moment,
                    no_prediction, models, subsample, n_threads, disable_avx, output, exe, logger,
                    vif_enhn_gain_limit=None, adm_enhn_gain_limit=None, motion_force_zero=False):

        if exe is None:
            exe = required(ExternalProgram.vmafexec)

        vmafexec_cmd = "{exe} --reference {reference} --distorted {distorted} --width {width} --height {height} " \
                     "--pixel_format {pixel_format} --bitdepth {bitdepth} --output {output}" \
            .format(
            exe=exe,
            reference=reference,
            distorted=distorted,
            width=width,
            height=height,
            pixel_format=pixel_format,
            bitdepth=bitdepth,
            output=output)

        if float_psnr:
            vmafexec_cmd += ' --feature float_psnr'
        if float_ssim:
            vmafexec_cmd += ' --feature float_ssim'
        if float_ms_ssim:
            vmafexec_cmd += ' --feature float_ms_ssim'
        if float_moment:
            vmafexec_cmd += ' --feature float_moment'

        if psnr:
            vmafexec_cmd += ' --feature psnr'
        if ssim:
            # vmafexec_cmd += ' --feature ssim'
            assert False, 'ssim (the daala integer ssim) is deprecated'
        if ms_ssim:
            vmafexec_cmd += ' --feature ms_ssim'

        if no_prediction:
            vmafexec_cmd += ' --no_prediction'
        else:
            assert models is not None
            assert isinstance(models, list)
            for model in models:
                vmafexec_cmd += ' --model {}'.format(model)

                # FIXME: hacky - since we do not know which feature is the one used in the model,
                # we have to set the parameter for all three, at the expense of extra computation.

                if vif_enhn_gain_limit is not None:
                    vmafexec_cmd += f':vif.vif_enhn_gain_limit={vif_enhn_gain_limit}:float_vif.vif_enhn_gain_limit={vif_enhn_gain_limit}'
                if adm_enhn_gain_limit is not None:
                    vmafexec_cmd += f':adm.adm_enhn_gain_limit={adm_enhn_gain_limit}:float_adm.adm_enhn_gain_limit={adm_enhn_gain_limit}'
                if motion_force_zero:
                    assert isinstance(motion_force_zero, bool)
                    motion_force_zero = str(motion_force_zero).lower()
                    vmafexec_cmd += f':motion.motion_force_zero={motion_force_zero}:float_motion.motion_force_zero={motion_force_zero}'

        assert isinstance(subsample, int) and subsample >= 1
        if subsample != 1:
            vmafexec_cmd += ' --subsample {}'.format(subsample)

        assert isinstance(n_threads, int) and n_threads >= 1
        if n_threads != 1:
            vmafexec_cmd += ' --threads {}'.format(n_threads)

        if disable_avx:
            vmafexec_cmd += ' --cpumask -1'

        if logger:
            logger.info(vmafexec_cmd)

        run_process(vmafexec_cmd, shell=True)


def model_path(*components):
    return os.path.join(VMAF_PYTHON_ROOT, "model", *components)
