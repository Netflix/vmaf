import os
import subprocess

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
VMAF_PROJECT = os.path.abspath(os.path.join(VMAF_LIB_FOLDER, '../../..',))


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


class ExternalProgram(object):
    """
    External C programs relied upon by the python vmaf code
    These external programs should be compiled before vmaf is ran, as per instructions in README
    """

    external_psnr = config.VmafExternalConfig.psnr_path()
    external_moment = config.VmafExternalConfig.moment_path()
    external_ssim = config.VmafExternalConfig.ssim_path()
    external_ms_ssim = config.VmafExternalConfig.ms_ssim_path()
    external_vmaf = config.VmafExternalConfig.vmaf_path()
    external_vmafossexec = config.VmafExternalConfig.vmafossexec_path()
    external_vmafrc = config.VmafExternalConfig.vmafrc_path()

    psnr = project_path(os.path.join("libvmaf", "build", "tools", "psnr")) if external_psnr is None else external_psnr
    moment = project_path(os.path.join("libvmaf", "build", "tools", "moment")) if external_moment is None else external_moment
    ssim = project_path(os.path.join("libvmaf", "build", "tools", "ssim")) if external_ssim is None else external_ssim
    ms_ssim = project_path(os.path.join("libvmaf", "build", "tools", "ms_ssim")) if external_ms_ssim is None else external_ms_ssim
    vmaf = project_path(os.path.join("libvmaf", "build", "tools", "vmaf")) if external_vmaf is None else external_vmaf
    vmafossexec = project_path(os.path.join("libvmaf", "build", "tools", "vmafossexec")) if external_vmafossexec is None else external_vmafossexec
    vmafrc = project_path(os.path.join("libvmaf", "build", "tools", "vmaf_rc")) if external_vmafrc is None else external_vmafrc


class ExternalProgramCaller(object):
    """
    Caller of ExternalProgram.
    """

    @staticmethod
    def call_psnr(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        psnr_cmd = "{psnr} {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            psnr=required(ExternalProgram.psnr),
            yuv_type=yuv_type,
            ref_path=ref_path,
            dis_path=dis_path,
            w=w,
            h=h,
            log_file_path=log_file_path,
        )
        if logger:
            logger.info(psnr_cmd)
        run_process(psnr_cmd, shell=True)

    @staticmethod
    def call_ssim(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        ssim_cmd = "{ssim} {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            ssim=required(ExternalProgram.ssim),
            yuv_type=yuv_type,
            ref_path=ref_path,
            dis_path=dis_path,
            w=w,
            h=h,
            log_file_path=log_file_path,
        )
        if logger:
            logger.info(ssim_cmd)
        run_process(ssim_cmd, shell=True)

    @staticmethod
    def call_ms_ssim(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        ms_ssim_cmd = "{ms_ssim} {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            ms_ssim=required(ExternalProgram.ms_ssim),
            yuv_type=yuv_type,
            ref_path=ref_path,
            dis_path=dis_path,
            w=w,
            h=h,
            log_file_path=log_file_path,
        )
        if logger:
            logger.info(ms_ssim_cmd)
        run_process(ms_ssim_cmd, shell=True)

    @staticmethod
    def call_vmaf_feature(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        vmaf_feature_cmd = "{vmaf} all {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            vmaf=required(ExternalProgram.vmaf),
            yuv_type=yuv_type,
            ref_path=ref_path,
            dis_path=dis_path,
            w=w,
            h=h,
            log_file_path=log_file_path,
        )
        if logger:
            logger.info(vmaf_feature_cmd)
        run_process(vmaf_feature_cmd, shell=True)

    @staticmethod
    def call_vifdiff_feature(yuv_type, ref_path, dis_path, w, h, log_file_path, logger=None):

        # APPEND (>>) result (since _prepare_generate_log_file method has already created the file
        # and written something in advance).
        vifdiff_feature_cmd = "{vmaf} vifdiff {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
            .format(
            vmaf=required(ExternalProgram.vmaf),
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
    def call_vmafrc(reference, distorted, width, height, pixel_format, bitdepth, float_psnr, psnr, float_ssim, ssim,
                    float_ms_ssim, ms_ssim, no_prediction, models, subsample, n_threads, disable_avx, output, exe,
                    logger):

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

        if logger:
            logger.info(vmafrc_cmd)

        run_process(vmafrc_cmd, shell=True)
