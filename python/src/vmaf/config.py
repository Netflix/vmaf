import os

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

PYTHON_ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(PYTHON_ROOT, '../../..',))

_MISSING_EXTERNAL_MESSAGE = """
Must install {name} and set {key} in %s/externals.py, e.g. add a line like
{key} = "[path to ffmpeg]/{name}"
""" % PYTHON_ROOT


def _path_from_external(name):
    """
    :param str name: Name of external configuration to look up
    :return str: Configured path, if any
    """
    try:
        import externals
        path = getattr(externals, name, None)
        if path and os.path.exists(path):
            return path

    except ImportError:
        pass

    return None


def ffmpeg_path():
    """
    :return str: Path to ffmpeg, if installed and configured via `externals` module
    """
    return _path_from_external('FFMPEG_PATH')


def matlab_path():
    """
    :return str: Path to matlab, if installed and configured via `externals` module
    """
    return _path_from_external('MATLAB_PATH')


def get_and_assert_ffmpeg():
    path = ffmpeg_path()
    assert path is not None, _MISSING_EXTERNAL_MESSAGE.format(name='ffmpeg', key='FFMPEG_PATH')
    return path


def get_and_assert_matlab():
    path = matlab_path()
    assert path is not None, _MISSING_EXTERNAL_MESSAGE.format(name='matlab', key='MATLAB_PATH')
    return path


# class VmafConfig(object):
#
#     @classmethod
#     def config_root_path(cls, *components):
#         return os.path.join(ROOT, *components)
#
#     @classmethod
#     def workspace_path(cls, *components):
#         return cls.config_root_path('workspace', *components)
#
#     @classmethod
#     def result_store_path(cls, *components):
#         return cls.workspace_path('result_store_dir', *components)
