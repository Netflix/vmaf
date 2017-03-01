import os


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
    moment = required_program("feature/moment")
    ms_ssim = required_program("feature/ms_ssim")
    psnr = required_program("feature/psnr")
    ssim = required_program("feature/ssim")
    vmaf = required_program("feature/vmaf")
    vmafossexec = required_program("wrapper/vmafossexec")
