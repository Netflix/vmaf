import os

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

PYTHON_ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(PYTHON_ROOT, '..',))

def get_and_assert_ffmpeg():
    try:
        import externals
        assert hasattr(externals, 'FFMPEG_PATH')
        assert os.path.exists(externals.FFMPEG_PATH)
        return externals.FFMPEG_PATH
    except (ImportError, AssertionError):
        msg = 'Must install ffmpeg and set FFMPEG_PATH in ' \
              '{python_path}/externals.py, e.g. add a line like\n' \
              'FFMPEG_PATH = "[path to ffmpeg]/ffmpeg"'.format(python_path=PYTHON_ROOT)
        raise AssertionError(msg)

def get_and_assert_matlab():
    try:
        import externals
        assert hasattr(externals, 'MATLAB_PATH')
        assert os.path.exists(externals.MATLAB_PATH)
        return externals.MATLAB_PATH
    except (ImportError, AssertionError):
        msg = 'Must install matlab and set MATLAB_PATH in ' \
              '{python_path}/externals.py, e.g. add a line like\n' \
              'MATLAB_PATH = "[path to matlab]/matlab"'.format(python_path=PYTHON_ROOT)
        raise AssertionError(msg)
