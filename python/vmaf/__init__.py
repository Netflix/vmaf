import os
import subprocess

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"
__version__ = "1.5.1"

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
VMAF_LIB_FOLDER = os.path.dirname(os.path.abspath(__file__))


# Assuming vmaf source checkout, path to top checked out folder
VMAF_PROJECT = os.path.abspath(os.path.join(VMAF_LIB_FOLDER, '../..',))


def run_process(cmd, **kwargs):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, **kwargs)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f'Process returned {e.returncode}, cmd: {cmd}, msg: {str(e.output)}')
    return 0


def project_path(relative_path):
    path = os.path.join(VMAF_PROJECT, relative_path)
    return path


def required(path):
    if not os.path.exists(path):
        raise AssertionError("%s does not exist, did you build?" % (path))
    return path


def convert_pixel_format_ffmpeg2vmafrc(ffmpeg_pix_fmt):
    '''
    Convert FFmpeg-style pixel format (pix_fmt) to vmaf_rc style.

    :param ffmpeg_pix_fmt: FFmpeg-style pixel format, for example: yuv420p, yuv420p10le
    :return: (pixel_format: str, bitdepth: int), for example: (420, 8), (420, 10)
    '''
    assert ffmpeg_pix_fmt in ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
    if ffmpeg_pix_fmt in ['yuv420p', 'yuv420p10le']:
        pixel_format = '420'
    elif ffmpeg_pix_fmt in ['yuv422p', 'yuv422p10le']:
        pixel_format = '422'
    elif ffmpeg_pix_fmt in ['yuv444p', 'yuv444p10le']:
        pixel_format = '444'
    else:
        assert False
    if ffmpeg_pix_fmt in ['yuv420p', 'yuv422p', 'yuv444p']:
        bitdepth = 8
    elif ffmpeg_pix_fmt in ['yuv420p10le', 'yuv422p10le', 'yuv444p10le']:
        bitdepth = 10
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
        external_vmafossexec = config.VmafExternalConfig.vmafossexec_path()
        external_vmafrc = config.VmafExternalConfig.vmafrc_path()
    except ImportError:
        external_vmaf_feature = None
        external_vmafossexec = None
        external_vmafrc = None

    vmaf_feature = project_path(os.path.join("libvmaf", "build", "tools", "vmaf_feature")) if external_vmaf_feature is None else external_vmaf_feature
    vmafossexec = project_path(os.path.join("libvmaf", "build", "tools", "vmafossexec")) if external_vmafossexec is None else external_vmafossexec
    vmafrc = project_path(os.path.join("libvmaf", "build", "tools", "vmaf_rc")) if external_vmafrc is None else external_vmafrc


class ExternalProgramCaller(object):
    """
    Caller of ExternalProgram.
    """

    @staticmethod
    def call_vmafrc_single_feature(feature, yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None, options=None):
        return ExternalProgramCaller.call_vmafrc_multi_features(
            [feature], yuv_type, ref_path, dis_path, w, h, log_file_path, logger=logger, options={feature: options})

    @staticmethod
    def call_vmafrc_multi_features(features, yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None, options=None):

        # ./libvmaf/build/tools/vmaf_rc
        # --reference python/test/resource/yuv/src01_hrc00_576x324.yuv
        # --distorted python/test/resource/yuv/src01_hrc01_576x324.yuv
        # --width 576 --height 324 --pixel_format 420 --bitdepth 8
        # --output /dev/stdout --xml --no_prediction --feature float_motion --feature integer_motion

        pixel_format, bitdepth = convert_pixel_format_ffmpeg2vmafrc(yuv_type)

        cmd = [
            required(ExternalProgram.vmafrc),
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

        for feature in features:
            if options is None:
                feature_str = feature
            else:
                assert isinstance(options, dict)
                if feature in options and options[feature] is not None:
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
        run_process(cmd)

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
    def call_vmafossexec(fmt, w, h, ref_path, dis_path, model, log_file_path, disable_clip_score,
                         enable_transform_score, phone_model, disable_avx, n_thread, n_subsample,
                         psnr, ssim, ms_ssim, ci, exe=None, logger=None):

        if exe is None:
            exe = required(ExternalProgram.vmafossexec)

        vmafossexec_cmd = "{exe} {fmt} {w} {h} {ref_path} {dis_path} {model} --log {log_file_path} --log-fmt xml --thread {n_thread} --subsample {n_subsample}" \
            .format(
            exe=exe,
            fmt=fmt,
            w=w,
            h=h,
            ref_path=ref_path,
            dis_path=dis_path,
            model=model,
            log_file_path=log_file_path,
            n_thread=n_thread,
            n_subsample=n_subsample)
        if disable_clip_score:
            vmafossexec_cmd += ' --disable-clip'
        if enable_transform_score or phone_model:
            vmafossexec_cmd += ' --enable-transform'
        if disable_avx:
            vmafossexec_cmd += ' --disable-avx'
        if psnr:
            vmafossexec_cmd += ' --psnr'
        if ssim:
            vmafossexec_cmd += ' --ssim'
        if ms_ssim:
            vmafossexec_cmd += ' --ms-ssim'
        if ci:
            vmafossexec_cmd += ' --ci'
        if logger:
            logger.info(vmafossexec_cmd)
        run_process(vmafossexec_cmd, shell=True)

    @staticmethod
    def call_vmafrc(reference, distorted, width, height, pixel_format, bitdepth,
                    float_psnr, psnr, float_ssim, ssim, float_ms_ssim, ms_ssim, float_moment,
                    no_prediction, models, subsample, n_threads, disable_avx, output, exe, logger,
                    vif_enhn_gain_limit=None, adm_enhn_gain_limit=None):

        if exe is None:
            exe = required(ExternalProgram.vmafrc)

        vmafrc_cmd = "{exe} --reference {reference} --distorted {distorted} --width {width} --height {height} " \
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
            vmafrc_cmd += ' --feature float_psnr'
        if float_ssim:
            vmafrc_cmd += ' --feature float_ssim'
        if float_ms_ssim:
            vmafrc_cmd += ' --feature float_ms_ssim'
        if float_moment:
            vmafrc_cmd += ' --feature float_moment'

        if psnr:
            vmafrc_cmd += ' --feature psnr'
        if ssim:
            vmafrc_cmd += ' --feature ssim'
        if ms_ssim:
            vmafrc_cmd += ' --feature ms_ssim'

        if no_prediction:
            vmafrc_cmd += ' --no_prediction'
        else:
            assert models is not None
            assert isinstance(models, list)
            for model in models:
                vmafrc_cmd += ' --model {}'.format(model)

        assert isinstance(subsample, int) and subsample >= 1
        if subsample != 1:
            vmafrc_cmd += ' --subsample {}'.format(subsample)

        assert isinstance(n_threads, int) and n_threads >= 1
        if n_threads != 1:
            vmafrc_cmd += ' --threads {}'.format(n_threads)

        if disable_avx:
            vmafrc_cmd += ' --cpumask -1'

        if vif_enhn_gain_limit is not None:
            vmafrc_cmd += f' --feature float_vif=vif_enhn_gain_limit={vif_enhn_gain_limit}'

        if adm_enhn_gain_limit is not None:
            vmafrc_cmd += f' --feature float_adm=adm_enhn_gain_limit={adm_enhn_gain_limit}'

        if logger:
            logger.info(vmafrc_cmd)

        run_process(vmafrc_cmd, shell=True)
