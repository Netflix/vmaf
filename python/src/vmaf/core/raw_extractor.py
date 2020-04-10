import os
from time import sleep

import numpy as np

from vmaf.core.h5py_mixin import H5pyMixin
from vmaf.tools.decorator import override
from vmaf.tools.reader import YuvReader
from vmaf.core.executor import Executor
from vmaf.core.result import RawResult

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

try:
    basestring
except NameError:
    # TODO: remove this once python2 support is dropped
    basestring = str


class RawExtractor(Executor):

    def _assert_args(self):
        super(RawExtractor, self)._assert_args()

        assert self.result_store is None, \
            "{} won't use result store.".format(self.__class__.__name__)


class AssetExtractor(RawExtractor):
    """
    AssetExtractor directly reads input assets and generate list of RawResult
    results that have the assets retrievable by result['asset']. The main
    purpose of this dummy extractor is to keep the interface uniform, when used
    by any pixel-domain TrainTestModel, such as a neural net.
    """

    TYPE = 'Asset'
    VERSION = '1.0'

    @classmethod
    @override(Executor)
    def _assert_an_asset(cls, asset):
        # override Executor._assert_an_asset bypassing ffmpeg check
        pass

    @override(Executor)
    def _open_ref_workfile(self, asset, fifo_mode):
        # do nothing
        pass

    @override(Executor)
    def _open_dis_workfile(self, asset, fifo_mode):
        # do nothing
        pass

    @override(Executor)
    def _wait_for_workfiles(self, asset):
        pass

    def _generate_result(self, asset):
        # do nothing
        pass

    def _read_result(self, asset):
        result = {
            'asset': asset
        }
        executor_id = self.executor_id
        return RawResult(asset, executor_id, result)


class DisYUVRawVideoExtractor(H5pyMixin, RawExtractor):
    """
    DisYUVRawVideoExtractor reads the distorted video Y, U, V channel into a
    h5py file
    """

    TYPE = 'DisYUVRawVideo'
    VERSION = '1.0'

    @property
    def channels(self):
        if self.optional_dict is None or 'channels' not in self.optional_dict:
            return 'yuv'
        else:
            channels = self.optional_dict['channels']
            assert isinstance(channels, basestring)
            channels = set(channels.lower())
            assert channels.issubset(set('yuv'))
            return ''.join(channels)

    @override(Executor)
    def run(self, **kwargs):
        if 'parallelize' in kwargs:
            parallelize = kwargs['parallelize']
        else:
            parallelize = False

        assert parallelize is False, "DisYUVRawVideoExtractor cannot parallelize."

        super(DisYUVRawVideoExtractor, self).run(**kwargs)

    def _assert_args(self):
        super(DisYUVRawVideoExtractor, self)._assert_args()
        self.assert_h5py_file()

    @override(Executor)
    def _open_ref_workfile(self, asset, fifo_mode):
        # do nothing
        pass

    @override(Executor)
    def _wait_for_workfiles(self, asset):
        # Override Executor._wait_for_workfiles to skip ref_workfile_path
        # wait til workfile paths being generated
        for i in range(10):
            if os.path.exists(asset.dis_workfile_path):
                break
            sleep(0.1)
        else:
            raise RuntimeError("dis video workfile path {} is missing.".format(
                asset.dis_workfile_path))

    def _generate_result(self, asset):
        quality_w, quality_h = asset.quality_width_height

        # count number of frames
        dis_ys = []
        dis_us = []
        dis_vs = []
        with YuvReader(filepath=asset.dis_procfile_path, width=quality_w, height=quality_h,
                       yuv_type=self._get_workfile_yuv_type(asset)) as dis_yuv_reader:
            for dis_yuv in dis_yuv_reader:
                dis_y, dis_u, dis_v = dis_yuv
                dis_y, dis_u, dis_v = dis_y.astype(np.double), dis_u.astype(np.double), dis_v.astype(np.double)

                dis_ys.append(dis_y)
                dis_us.append(dis_u)
                dis_vs.append(dis_v)

        # Y
        if 'y' in self.channels.lower():
            h5py_cache_y = self.h5py_file.create_dataset(
                str(asset)+'_y',
                (len(dis_ys), dis_ys[0].shape[0], dis_ys[0].shape[1]),
                dtype='float'
            )
            h5py_cache_y.dims[0].label = 'frame'
            h5py_cache_y.dims[1].label = 'height'
            h5py_cache_y.dims[2].label = 'width'
            for idx, dis_y in enumerate(dis_ys):
                h5py_cache_y[idx] = dis_y

        # U
        if 'u' in self.channels.lower():
            h5py_cache_u = self.h5py_file.create_dataset(
                str(asset)+'_u',
                (len(dis_us), dis_us[0].shape[0], dis_us[0].shape[1]),
                dtype='float'
            )
            h5py_cache_u.dims[0].label = 'frame'
            h5py_cache_u.dims[1].label = 'height'
            h5py_cache_u.dims[2].label = 'width'
            for idx, dis_u in enumerate(dis_us):
                h5py_cache_u[idx] = dis_u

        # V
        if 'v' in self.channels.lower():
            h5py_cache_v = self.h5py_file.create_dataset(
                str(asset)+'_v',
                (len(dis_vs), dis_vs[0].shape[0], dis_vs[0].shape[1]),
                dtype='float'
            )
            h5py_cache_v.dims[0].label = 'frame'
            h5py_cache_v.dims[1].label = 'height'
            h5py_cache_v.dims[2].label = 'width'
            for idx, dis_v in enumerate(dis_vs):
                h5py_cache_v[idx] = dis_v

    def _read_result(self, asset):
        result = {}
        if 'y' in self.channels.lower():
            result['dis_y'] = self.h5py_file[str(asset)+'_y']
        if 'u' in self.channels.lower():
            result['dis_u'] = self.h5py_file[str(asset)+'_u']
        if 'v' in self.channels.lower():
            result['dis_v'] = self.h5py_file[str(asset)+'_v']

        executor_id = self.executor_id
        return RawResult(asset, executor_id, result)
