__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import sys
import os

def get_stdout_logger():
    import logging
    logger = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def get_file_name_without_extension(f):
    """

    >>> get_file_name_without_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('yuv/src01_hrc01')
    'src01_hrc01'

    """
    return os.path.splitext(f.split("/")[-1])[0]
