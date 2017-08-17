import os

from vmaf.tools.misc import run_process

# Path to folder containing this file
VMAF_LIB_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Assuming vmaf source checkout, path to top checked out folder
VMAF_PROJECT = os.path.abspath(os.path.join(VMAF_LIB_FOLDER, '../../..',))


def project_path(relative_path, required=None):
    """
    :param str relative_path: Path relative to vmaf project source tree
    :param str|None required: Optional, if provided raise an exception when file at 'relative_path' is missing
    :return str: Full path to program
    """
    path = os.path.join(VMAF_PROJECT, relative_path)
    if required and not os.path.exists(path):
        raise Exception("%s does not exist %s" % (path, required))
    return path


def required_program(relative_path):
    return project_path(relative_path, required="did you build?")


class ExternalProgram(object):
    """
    External C programs relied upon by the python vmaf code
    These external programs should be compiled before vmaf is ran, as per instructions in README
    """
    psnr = required_program("feature/psnr")
    moment = required_program("feature/moment")
    ssim = required_program("feature/ssim")
    ms_ssim = required_program("feature/ms_ssim")
    vmaf = required_program("feature/vmaf")
    vmafossexec = required_program("wrapper/vmafossexec")

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
            psnr=ExternalProgram.psnr,
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
            ssim=ExternalProgram.ssim,
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
            ms_ssim=ExternalProgram.ms_ssim,
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
            vmaf=ExternalProgram.vmaf,
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
    def call_vmafossexec(fmt, w, h, ref_path, dis_path, model, log_file_path, disable_clip_score,
                         enable_transform_score, phone_model, disable_avx, exe=None, logger=None):

        if exe is None:
            exe = ExternalProgram.vmafossexec

        vmafossexec_cmd = "{exe} {fmt} {w} {h} {ref_path} {dis_path} {model} --log {log_file_path} --log-fmt xml --psnr --ssim --ms-ssim" \
            .format(
            exe=exe,
            fmt=fmt,
            w=w,
            h=h,
            ref_path=ref_path,
            dis_path=dis_path,
            model=model,
            log_file_path=log_file_path,
        )
        if disable_clip_score:
            vmafossexec_cmd += ' --disable-clip'
        if enable_transform_score or phone_model:
            vmafossexec_cmd += ' --enable-transform'
        if disable_avx:
            vmafossexec_cmd += ' --disable-avx'
        if logger:
            logger.info(vmafossexec_cmd)
        run_process(vmafossexec_cmd, shell=True)
