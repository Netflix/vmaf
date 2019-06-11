from fnmatch import fnmatch
import multiprocessing
from time import sleep, time
import itertools

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import errno
import os
import re

from vmaf import run_process, to_list
from vmaf.tools.scanf import sscanf, IncompleteCaptureError, FormatError

try:
    unicode  # noqa, remove this once python2 support is dropped

except NameError:
    unicode = str


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


def get_file_name_without_extension(path):
    """

    >>> get_file_name_without_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('yuv/src01_hrc01')
    'src01_hrc01'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.sdr.yuv')
    'src01_hrc01.sdr'
    >>> get_file_name_without_extension('abc/xyz/src01_hrc01.sdr.dvi.yuv')
    'src01_hrc01.sdr.dvi'

    """
    return os.path.splitext(path.split("/")[-1])[0]


def get_file_name_with_extension(path):
    """

    >>> get_file_name_with_extension('yuv/src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('src01_hrc01.yuv')
    'src01_hrc01.yuv'
    >>> get_file_name_with_extension('abc/xyz/src01_hrc01.yuv')
    'src01_hrc01.yuv'

    """
    return path.split("/")[-1]


def get_file_name_extension(path):
    '''
    >>> get_file_name_extension("file:///mnt/zli/test.txt")
    'txt'
    >>> get_file_name_extension("test.txt")
    'txt'
    >>> get_file_name_extension("abc")
    'abc'
    '''
    return path.split('.')[-1]


def get_normalized_path(dir_):
    """

    >>> get_normalized_path('abc/xyz/')
    'abc/xyz'
    >>> get_normalized_path('abc/xyz')
    'abc/xyz'
    >>> get_normalized_path('abc/xyz.txt')
    'abc/xyz.txt'

    """
    if dir_[-1] == '/':
        return dir_[:-1]
    else:
        return dir_


def get_dir_without_last_slash(path):
    """

    >>> get_dir_without_last_slash('abc/src01_hrc01.yuv')
    'abc'
    >>> get_dir_without_last_slash('src01_hrc01.yuv')
    ''
    >>> get_dir_without_last_slash('abc/xyz/src01_hrc01.yuv')
    'abc/xyz'
    >>> get_dir_without_last_slash('abc/xyz/')
    'abc/xyz'

    """
    return "/".join(path.split("/")[:-1])


def make_parent_dirs_if_nonexist(path):
    dst_dir = get_dir_without_last_slash(path)
    # create dir if not exist yet
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def delete_dir_if_exists(dir):
    if os.path.isdir(dir):
        os.rmdir(dir)


def get_normalized_string_from_dict(d):
    """ Normalized string representation with sorted keys.

    >>> get_normalized_string_from_dict({"max_buffer_sec": 5.0, "bitrate_kbps": 45, })
    'bitrate_kbps_45_max_buffer_sec_5.0'
    """
    return '_'.join(map(lambda k: '{k}_{v}'.format(k=k,v=d[k]), sorted(d.keys())))


def get_hashable_value_tuple_from_dict(d):
    """ Hashable tuple of values with sorted keys.

    >>> get_hashable_value_tuple_from_dict({"max_buffer_sec": 5.0, "bitrate_kbps": 45, })
    (45, 5.0)
    >>> get_hashable_value_tuple_from_dict({"max_buffer_sec": 5.0, "bitrate_kbps": 45, "resolutions": [(740, 480), (1920, 1080), ]})
    (45, 5.0, ((740, 480), (1920, 1080)))
    """
    return tuple(map(
        lambda k: tuple(d[k]) if isinstance(d[k], list) else d[k],
        sorted(d.keys())))


def get_unique_str_from_recursive_dict(d):
    """ String representation with sorted keys and values for recursive dict.

    >>> get_unique_str_from_recursive_dict({'a':1, 'b':2, 'c':{'x':'0', 'y':'1'}})
    '{"a": 1, "b": 2, "c": {"x": "0", "y": "1"}}'
    >>> get_unique_str_from_recursive_dict({'a':1, 'c':2, 'b':{'y':'1', 'x':'0', }})
    '{"a": 1, "b": {"x": "0", "y": "1"}, "c": 2}'
    """
    from collections import OrderedDict
    import json

    def to_ordered_dict_recursively(d):
        if isinstance(d, dict):
            return OrderedDict(map(
                lambda t: (to_ordered_dict_recursively(t[0]), to_ordered_dict_recursively(t[1])),
                sorted(d.items())
            ))
        else:
            return d
    return json.dumps(to_ordered_dict_recursively(d))


def indices(a, func):
    """
    Get indices of elements in an array which satisfies func
    >>> indices([1, 2, 3, 4], lambda x: x>2)
    [2, 3]
    >>> indices([1, 2, 3, 4], lambda x: x==2.5)
    []
    >>> indices([1, 2, 3, 4], lambda x: x>1 and x<=3)
    [1, 2]
    >>> indices([1, 2, 3, 4], lambda x: x in [2, 4])
    [1, 3]
    >>> indices([1,2,3,1,2,3,1,2,3], lambda x: x > 2)
    [2, 5, 8]
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def import_python_file(filepath):
    """
    Import a python file as a module.
    :param filepath:
    :return:
    """
    import imp
    filename = get_file_name_without_extension(filepath)
    ret = imp.load_source(filename, filepath)

    return ret


def make_absolute_path(path, current_dir):
    '''

    >>> make_absolute_path('abc/cde.fg', '/xyz/')
    '/xyz/abc/cde.fg'
    >>> make_absolute_path('/abc/cde.fg', '/xyz/')
    '/abc/cde.fg'

    '''
    if path[0] == '/':
        return path
    else:
        return current_dir + path


def empty_object():
    return type('', (), {})()


def get_cmd_option(argv, begin, end, option):
    '''

    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 3, 5, '--xyz')
    '123'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, '--xyz')
    '123'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 4, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 5, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 6, 5, '--xyz')
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, 'a')
    'b'
    >>> get_cmd_option(['a', 'b', 'c', '--xyz', '123'], 0, 5, 'b')
    'c'

    '''
    itr = None
    for itr in range(begin, end):
        if argv[itr] == option:
            break
    if itr is not None and itr != end and (itr + 1) != end:
        return argv[itr + 1]
    return None


def cmd_option_exists(argv, begin, end, option):
    '''

    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'c')
    True
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 3, 4, 'c')
    False
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 3, 4, 'd')
    True
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'a')
    False
    >>> cmd_option_exists(['a', 'b', 'c', 'd'], 2, 4, 'b')
    False

    '''
    found = False
    for itr in range(begin, end):
        if argv[itr] == option:
            found = True
            break
    return found


def index_and_value_of_min(l):
    '''

    >>> index_and_value_of_min([2, 0, 3])
    (1, 0)

    '''
    return min(enumerate(l), key=lambda x: x[1])


def parallel_map(func, list_args, processes=None):
    """
    Build my own parallelized map function since multiprocessing's Process(),
    or Pool.map() cannot meet my both needs:
    1) be able to control the maximum number of processes in parallel
    2) be able to take in non-picklable objects as arguments
    """

    # get maximum number of active processes that can be used
    max_active_procs = processes if processes is not None else multiprocessing.cpu_count()

    # create shared dictionary
    return_dict = multiprocessing.Manager().dict()

    # define runner function
    def func_wrapper(idx_args):
        idx, args = idx_args
        executor = func(args)
        return_dict[idx] = executor

    # add idx to args
    list_idx_args = []
    for idx, args in enumerate(list_args):
        list_idx_args.append((idx, args))

    procs = []
    for idx_args in list_idx_args:
        proc = multiprocessing.Process(target=func_wrapper, args=(idx_args,))
        procs.append(proc)

    waiting_procs = set(procs)
    active_procs = set([])

    # processing
    while True:

        # check if any procs in active_procs is done; if yes, remove them
        for p in active_procs.copy():
            if not p.is_alive():
                active_procs.remove(p)

        # check if can add a proc to active_procs (add gradually one per loop)
        if len(active_procs) < max_active_procs and len(waiting_procs) > 0:
            # move one proc from waiting_procs to active_procs
            p = waiting_procs.pop()
            active_procs.add(p)
            p.start()

        # if both waiting_procs and active_procs are empty, can terminate
        if len(waiting_procs) == 0 and len(active_procs) == 0:
            break

        sleep(0.01) # check every x sec

    # finally, collect results
    rets = map(lambda idx: return_dict[idx], range(len(list_args)))
    return to_list(rets)


def check_program_exist(program):
    '''

    >>> check_program_exist("xxxafasd34df")
    False
    >>> check_program_exist("xxxafasd34df f899")
    False
    >>> check_program_exist("ls")
    True
    >>> check_program_exist("ls -all")
    True
    >>> check_program_exist("pwd")
    True

    '''
    try:
        run_process(program.split(), stdout=open(os.devnull, 'wb'))
        return True
    except OSError as e:
        if e.errno == errno.ENOENT:
            return False
        else:
            # Something else went wrong while trying to run `wget`
            raise


def check_scanf_match(string, template):
    '''
    >>> check_scanf_match('frame00000000.icpf', 'frame%08d.icpf')
    True
    >>> check_scanf_match('frame00000003.icpf', 'frame%08d.icpf')
    True
    >>> check_scanf_match('frame0000001.icpf', 'frame%08d.icpf')
    True
    >>> check_scanf_match('frame00000001.icpff', 'frame%08d.icpf')
    True
    >>> check_scanf_match('gframe00000001.icpff', 'frame%08d.icpf')
    False
    >>> check_scanf_match('fyrame00000001.icpff', 'frame%08d.icpf')
    False
    >>> check_scanf_match('xx/yy/frame00000000.icpf', 'xx/yy/frame%08d.icpf')
    True
    >>> check_scanf_match('xx/yy//frame00000000.icpf', 'xx/yy/frame%08d.icpf')
    False
    >>> check_scanf_match('xx/yy/frame00000000.icpf', 'xx/yy//frame%08d.icpf')
    False
    >>> check_scanf_match("-1-2+3-4", "%02d%02d%02d%02d")
    True
    >>> check_scanf_match('frame00000240.icpf', 'frame%08d.icpf')
    True
    >>> check_scanf_match('/mnt/hgfs/ZLI-NFLX-10/USCJND/ref/1920x1080/videoSRC001_1920x1080_30.yuv.avi', '/mnt/hgfs/ZLI-NFLX-10/USCJND/ref/1920x1080/videoSRC001_1920x1080_*.yuv.avi')
    True
    '''
    ret = False
    try:
        sscanf(string, template)
        return True
    except (FormatError, IncompleteCaptureError):
        pass

    if fnmatch(string, template):
        return True

    return False


def match_any_files(template):
    dir_ = os.path.dirname(template)
    for filename in os.listdir(dir_):
        filepath = dir_ + '/' + filename
        if check_scanf_match(filepath, template):
            return True
    return False


def unroll_dict_of_lists(dict_of_lists):
    """ Unfold a dictionary of lists into a list of dictionaries.

    >>> dict_of_lists = {'norm_type':['normalize'], 'n_estimators':[10, 50], 'random_state': [0]}
    >>> expected = [{'n_estimators': 10, 'norm_type': 'normalize', 'random_state': 0}, {'n_estimators': 50, 'norm_type': 'normalize', 'random_state': 0}]
    >>> unroll_dict_of_lists(dict_of_lists) == expected
    True

    """
    keys = sorted(dict_of_lists.keys()) # normalize order
    list_of_key_value_pairs = []
    for key in keys:
        values = dict_of_lists[key]
        key_value_pairs = []
        for value in values:
            key_value_pairs.append((key, value))
        list_of_key_value_pairs.append(key_value_pairs)

    list_of_key_value_pairs_rearranged = \
        itertools.product(*list_of_key_value_pairs)

    list_of_dicts = []
    for key_value_pairs in list_of_key_value_pairs_rearranged:
        list_of_dicts.append(dict(key_value_pairs))

    return list_of_dicts


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = unicodedata.normalize('NFKD', unicode(value)).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))

    return value


def neg_if_even(x):
    """
    >>> neg_if_even(2)
    -1
    >>> neg_if_even(1)
    1
    >>> neg_if_even(0)
    -1
    >>> neg_if_even(-1)
    1
    >>> neg_if_even(-2)
    -1

    """
    return 1 - (x % 2 == 0) * 2


def get_unique_sorted_list(l):
    """
    >>> get_unique_sorted_list([3, 4, 4, 1])
    [1, 3, 4]
    >>> get_unique_sorted_list([])
    []
    """
    return sorted(list(set(l)))


class Timer(object):

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        print('Elapsed: %s' % (time() - self.tstart))


def dedup_value_in_dict(d):
    """
    >>> dedup_value_in_dict({'a': 1, 'b': 1, 'c': 2}) == {'a': 1, 'c': 2}
    True
    """
    reversed_d = dict()
    keys = sorted(d.keys())
    for key in keys:
        value = d[key]
        if value not in reversed_d:
            reversed_d[value] = key
    d_ = dict()
    for value, key in reversed_d.items():
        d_[key] = value
    return d_


if __name__ == '__main__':
    import doctest
    doctest.testmod()
