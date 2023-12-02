import os
import hashlib
import ast
import shutil

import pandas as pd

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.result import Result
from vmaf.tools.misc import make_parent_dirs_if_nonexist

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

class ResultStore(object):
    """
    Provide capability to save and load a Result.
    """
    pass


class SqliteResultStore(ResultStore):
    """
    persist result by a SQLite engine that save/load result.
    """
    pass


class FileSystemResultStore(ResultStore):
    """
    persist result by a simple file system that save/load result in a directory.
    The directory has multiple subdirectories, each corresponding to an Executor
    (e.g. a VMAF feature extractor, or a NO19 feature extractor, or a
    VMAF quality runner, or a SSIM quality runner). Each subdirectory contains
    multiple files, each file stores dataframe for an asset, and has file name
    str(asset).
    """
    def __init__(self, logger=None,
                 result_store_dir=VmafConfig.file_result_store_path()
                 ):
        self.logger = logger
        self.result_store_dir = result_store_dir

    def save(self, result):
        result_file_path = self._get_result_file_path(result)
        try:
            make_parent_dirs_if_nonexist(result_file_path)
        except OSError as e:
            print('make_parent_dirs_if_nonexist {path} fails: {e}'.format(path=result_file_path, e=str(e)))

        self.save_result(result, result_file_path)

    def save_workfile(self, result: Result, workfile_path: str, suffix: str):
        result_file_path = self._get_result_file_path(result)
        try:
            make_parent_dirs_if_nonexist(result_file_path)
        except OSError as e:
            print('make_parent_dirs_if_nonexist {path} fails: {e}'.format(path=result_file_path, e=str(e)))

        shutil.copyfile(workfile_path, result_file_path + suffix)

    def load(self, asset, executor_id):
        result_file_path = self._get_result_file_path2(asset, executor_id)
        if not os.path.isfile(result_file_path):
            return None
        result = self.load_result(result_file_path, asset.__class__)
        return result

    def has_workfile(self, asset: Asset, executor_id: str, suffix: str) -> bool:
        result_file_path = self._get_result_file_path2(asset, executor_id)
        return os.path.isfile(result_file_path + suffix)

    @staticmethod
    def save_result(result, result_file_path):
        with open(result_file_path, "wt") as result_file:
            result_file.write(str(result.to_dataframe().to_dict()))

    @staticmethod
    def load_result(result_file_path, AssetClass=Asset):
        with open(result_file_path, "rt") as result_file:
            df = pd.DataFrame.from_dict(ast.literal_eval(result_file.read()))
            result = Result.from_dataframe(df, AssetClass)
        return result

    def delete(self, asset, executor_id):
        result_file_path = self._get_result_file_path2(asset, executor_id)
        if os.path.isfile(result_file_path):
            os.remove(result_file_path)

    def delete_workfile(self, asset, executor_id, suffix: str):
        result_file_path = self._get_result_file_path2(asset, executor_id)
        workfile_path = result_file_path + suffix
        if os.path.isfile(workfile_path):
            os.remove(workfile_path)

    def clean_up(self):
        """
        WARNING: RMOVE ENTIRE RESULT STORE, USE WITH CAUTION!!!
        :return:
        """
        import shutil
        if os.path.isdir(self.result_store_dir):
            shutil.rmtree(self.result_store_dir)

    def _get_result_file_path(self, result):
        return self._get_result_file_path2(result.asset, result.executor_id)

    def _get_result_file_path2(self, asset, executor_id):
        str_to_hash = str(asset).encode("utf-8")
        return "{dir}/{executor_id}/{dataset}/{content_id}/{str}".format(
            dir=self.result_store_dir, executor_id=executor_id,
            dataset=asset.dataset,
            content_id=asset.content_id,
            str=hashlib.sha1(str_to_hash).hexdigest())
