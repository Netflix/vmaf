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


# Path to folder containing this file
VMAF_LIB_FOLDER = os.path.dirname(os.path.abspath(__file__))


# Assuming vmaf source checkout, path to top checked out folder
VMAF_PROJECT = os.path.abspath(os.path.join(VMAF_LIB_FOLDER, '../../..',))


def to_list(value):
    """
    TODO python3: map() and filter() return a generator in python3, however vmaf assumes list

    Returns:
        (list): `value` converted to a list when applicable
    """
    if value is None or isinstance(value, list):
        return value
    return list(value)


def run_process(cmd, **kwargs):
    ret = subprocess.call(cmd, **kwargs)
    assert ret == 0, 'Process returned {ret}, cmd: {cmd}'.format(ret=ret, cmd=cmd)
    return ret


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
    psnr = project_path(os.path.join("src", "libvmaf", "psnr"))
    moment = project_path(os.path.join("src", "libvmaf", "moment"))
    ssim = project_path(os.path.join("src", "libvmaf", "ssim"))
    ms_ssim = project_path(os.path.join("src", "libvmaf", "ms_ssim"))
    vmaf = project_path(os.path.join("src", "libvmaf", "vmaf"))
    vmafossexec = project_path(os.path.join("src", "libvmaf", "vmafossexec"))


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
