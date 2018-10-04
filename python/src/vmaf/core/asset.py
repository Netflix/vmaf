import copy
from vmaf.tools.decorator import deprecated

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os

from vmaf.core.mixin import WorkdirEnabled
from vmaf.tools.misc import get_file_name_without_extension, \
    get_file_name_with_extension, get_unique_str_from_recursive_dict
from vmaf.config import VmafConfig


class Asset(WorkdirEnabled):
    """
    An Asset is the most basic unit with sufficient information to perform an
    execution task. It includes basic information about a distorted video and
    its undistorted reference video, as well as the frame range on which to
    extract features/calculate quality results (*dis_start_end_frame* and
    *ref_start_end_frame*), and at what resolution to perform such feature
    extraction (each video frame is upscaled to the resolution specified by
    *quality_width_hight* before processing).

    Asset extends WorkdirEnabled mixin, which comes with a thread-safe working
    directory to facilitate parallel execution.

    The ref_path/dis_path points to the reference/distorted video files. For now,
    it supports YUV video files (yuvxxx), or encoded video files (notyuv) that
    can be decoded by ffmpeg.
    """

    SUPPORTED_YUV_TYPES = ['yuv420p', 'yuv422p', 'yuv444p',
                           'yuv420p10le', 'yuv422p10le', 'yuv444p10le',
                           'notyuv']
    DEFAULT_YUV_TYPE = 'yuv420p'

    SUPPORTED_RESAMPLING_TYPES = ['bilinear', 'bicubic', 'lanczos']
    DEFAULT_RESAMPLING_TYPE = 'bicubic'

    # ==== constructor ====

    def __init__(self, dataset, content_id, asset_id,
                 ref_path, dis_path,
                 asset_dict,
                 workdir_root=VmafConfig.workdir_path()):
        """
        :param dataset
        :param content_id: ID of content the asset correspond to within dataset
        :param asset_id: ID of asset
        :param ref_path: path to reference video
        :param dis_path: path to distorted video
        :param asset_dict: dictionary with additional asset properties
        :param workdir_root:
        :return:
        """
        WorkdirEnabled.__init__(self, workdir_root)
        self.dataset = dataset
        self.content_id = content_id
        self.asset_id = asset_id
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.asset_dict = asset_dict

        self._assert()

    def _assert(self):
        # validate yuv types
        assert self.ref_yuv_type in self.SUPPORTED_YUV_TYPES
        assert self.dis_yuv_type in self.SUPPORTED_YUV_TYPES
        assert self.workfile_yuv_type in self.SUPPORTED_YUV_TYPES
        # if YUV is notyuv, then ref/dis width and height should not be given,
        # since it must be encoded video and the information should be already
        # in included in the header of the video files
        if self.ref_yuv_type == 'notyuv':
            assert self.ref_width_height is None, 'For ref_yuv_type nonyuv, ref_width_height must NOT be specified.'
        if self.dis_yuv_type == 'notyuv':
            assert self.dis_width_height is None, 'For dis_yuv_type nonyuv, dis_width_height must NOT be specified.'

    def copy(self, **kwargs):
        new_asset_dict = copy.deepcopy(self.asset_dict)

        # reset the following argument:
        if 'use_path_as_workpath' in new_asset_dict:
            del new_asset_dict['use_path_as_workpath']

        dataset = kwargs['dataset'] if 'dataset' in kwargs else self.dataset
        content_id = kwargs['content_id'] if 'content_id' in kwargs else self.content_id
        asset_id = kwargs['asset_id'] if 'asset_id' in kwargs else self.asset_id
        ref_path = kwargs['ref_path'] if 'ref_path' in kwargs else self.ref_path
        dis_path = kwargs['dis_path'] if 'dis_path' in kwargs else self.dis_path
        workdir_root = kwargs['workdir_root'] if 'workdir_root' in kwargs else self.workdir_root

        new_asset = self.__class__(dataset, content_id, asset_id,
                                   ref_path, dis_path, new_asset_dict,
                                   workdir_root)
        return new_asset

    @staticmethod
    def from_repr(rp):
        """
        Reconstruct Asset from repr string.
        :return:
        """
        import ast
        d = ast.literal_eval(rp)
        assert 'dataset' in d
        assert 'content_id' in d
        assert 'asset_id' in d
        assert 'ref_path' in d
        assert 'dis_path' in d
        assert 'asset_dict' in d

        return Asset(dataset=d['dataset'],
                     content_id=d['content_id'],
                     asset_id=d['asset_id'],
                     ref_path=d['ref_path'],
                     dis_path=d['dis_path'],
                     asset_dict=d['asset_dict']
                     )

    # ==== groundtruth ====
    @property
    def groundtruth(self):
        """
        Ground truth score, e.g. MOS, DMOS
        :return:
        """
        if 'groundtruth' in self.asset_dict:
            return self.asset_dict['groundtruth']
        else:
            return None

    @property
    def groundtruth_std(self):
        if 'groundtruth_std' in self.asset_dict:
            return self.asset_dict['groundtruth_std']
        else:
            return None

    @property
    def raw_groundtruth(self):
        """
        Raw ground truth scores, e.g. opinion score (OS)
        :return:
        """
        if 'raw_groundtruth' in self.asset_dict:
            return self.asset_dict['raw_groundtruth']
        else:
            return None

    # ==== width and height ====

    @property
    def ref_width_height(self):
        """
        Width and height of reference video.
        :return: width and height of reference video. If None, it signals that
        width and height should be figured out in other means (e.g. FFMPEG).
        """
        if 'ref_width' in self.asset_dict and 'ref_height' in self.asset_dict:
            return self.asset_dict['ref_width'], self.asset_dict['ref_height']
        elif 'width' in self.asset_dict and 'height' in self.asset_dict:
            return self.asset_dict['width'], self.asset_dict['height']
        else:
            return None

    @property
    def dis_width_height(self):
        """
        Width and height of distorted video.
        :return: width and height of distorted video. If None, it signals that
        width and height should be figured out in other means (e.g. FFMPEG)
        """
        if 'dis_width' in self.asset_dict and 'dis_height' in self.asset_dict:
            return self.asset_dict['dis_width'], self.asset_dict['dis_height']
        elif 'width' in self.asset_dict and 'height' in self.asset_dict:
            return self.asset_dict['width'], self.asset_dict['height']
        else:
            return None

    def clear_up_width_height(self):
        if 'width' in self.asset_dict:
            del self.asset_dict['width']
        if 'height' in self.asset_dict:
            del self.asset_dict['height']
        if 'ref_width' in self.asset_dict:
            del self.asset_dict['ref_width']
        if 'ref_height' in self.asset_dict:
            del self.asset_dict['ref_height']
        if 'dis_width' in self.asset_dict:
            del self.asset_dict['dis_width']
        if 'dis_height' in self.asset_dict:
            del self.asset_dict['dis_height']

    @property
    def quality_width_height(self):
        """
        Width and height to scale distorted video to before quality calculation.
        :return: width and height at which the quality is measured at. either
        'quality_width' and 'quality_height' have to present in asset_dict;
        or ref and dis's width and height must be equal, which will be used
        as the default quality width and height; or either of ref/dis is type
        'notyuv', in which case the other's width/height (could also be None)
        """

        if 'quality_width' in self.asset_dict and 'quality_height' in self.asset_dict:
            return self.asset_dict['quality_width'], self.asset_dict['quality_height']
        elif self.ref_yuv_type == 'notyuv':
            return self.dis_width_height
        elif self.dis_yuv_type == 'notyuv':
            return self.ref_width_height
        else:
            assert self.ref_width_height == self.dis_width_height
            return self.ref_width_height

    # ==== start and end frame ====

    @property
    def ref_start_end_frame(self):
        """
        Start and end frame of reference video for quality calculation.
        :return: reference video's start frame and end frame for processing
        (inclusive). If None, it signals that the entire video should be
        processed.
        """
        if 'ref_start_frame' in self.asset_dict and 'ref_end_frame' in self.asset_dict:
            return self.asset_dict['ref_start_frame'], self.asset_dict['ref_end_frame']

        elif 'start_frame' in self.asset_dict and 'end_frame' in self.asset_dict:
            return self.asset_dict['start_frame'], self.asset_dict['end_frame']

        elif 'start_sec' in self.asset_dict and 'end_sec' in self.asset_dict and 'fps' in self.asset_dict:
            start_frame = int(round(self.asset_dict['start_sec'] * self.asset_dict['fps']))
            end_frame = int(round(self.asset_dict['end_sec'] * self.asset_dict['fps'])) - 1
            return start_frame, end_frame

        elif 'duration_sec' in self.asset_dict and 'fps' in self.asset_dict:
            start_frame = 0
            end_frame = int(round(self.asset_dict['duration_sec'] * self.asset_dict['fps'])) - 1
            return start_frame, end_frame

        else:
            return None

    @property
    def dis_start_end_frame(self):
        """
        Start and end frame of distorted video for quality calculation.
        :return: distorted video's start frame and end frame for processing
        (inclusive). If None, it signals that the entire video should be
        processed.
        """
        if 'dis_start_frame' in self.asset_dict and 'dis_end_frame' in self.asset_dict:
            return self.asset_dict['dis_start_frame'], self.asset_dict['dis_end_frame']

        elif 'start_frame' in self.asset_dict and 'end_frame' in self.asset_dict:
            return self.asset_dict['start_frame'], self.asset_dict['end_frame']

        elif 'start_sec' in self.asset_dict and 'end_sec' in self.asset_dict and 'fps' in self.asset_dict:
            start_frame = int(round(self.asset_dict['start_sec'] * self.asset_dict['fps']))
            end_frame = int(round(self.asset_dict['end_sec'] * self.asset_dict['fps'])) - 1
            return start_frame, end_frame

        elif 'duration_sec' in self.asset_dict and 'fps' in self.asset_dict:
            start_frame = 0
            end_frame = int(round(self.asset_dict['duration_sec'] * self.asset_dict['fps'])) - 1
            return start_frame, end_frame

        else:
            return None

    def clear_up_start_end_frame(self):
        if 'start_frame' in self.asset_dict:
            del self.asset_dict['start_frame']
        if 'end_frame' in self.asset_dict:
            del self.asset_dict['end_frame']
        if 'ref_start_frame' in self.asset_dict:
            del self.asset_dict['ref_start_frame']
        if 'dis_start_frame' in self.asset_dict:
            del self.asset_dict['dis_start_frame']
        if 'start_sec' in self.asset_dict:
            del self.asset_dict['start_sec']
        if 'end_sec' in self.asset_dict:
            del self.asset_dict['end_sec']
        if 'duration_sec' in self.asset_dict:
            del self.asset_dict['duration_sec']

    # ==== duration and start time====

    @property
    def ref_duration_sec(self):
        """
        Reference video's duration in second used in quality calculation.
        :return:
        """
        if 'duration_sec' in self.asset_dict:
            return self.asset_dict['duration_sec']
        elif 'start_sec' in self.asset_dict \
                and 'end_sec' in self.asset_dict:
            return self.asset_dict['end_sec'] - self.asset_dict['start_sec']
        else:
            ref_start_end_frame = self.ref_start_end_frame
            if ref_start_end_frame and 'fps' in self.asset_dict:
                s, e = ref_start_end_frame
                return (e - s + 1) / float(self.asset_dict['fps'])
            else:
                return None

    @property
    def dis_duration_sec(self):
        """
        Distorted video's duration in second used in quality calculation.
        :return:
        """
        if 'duration_sec' in self.asset_dict:
            return self.asset_dict['duration_sec']
        elif 'start_sec' in self.asset_dict \
                and 'end_sec' in self.asset_dict:
            return self.asset_dict['end_sec'] - self.asset_dict['start_sec']
        else:
            dis_start_end_frame = self.dis_start_end_frame
            if dis_start_end_frame \
                    and 'fps' in self.asset_dict:
                start, end = dis_start_end_frame
                return (end - start + 1) / float(self.asset_dict['fps'])
            else:
                return None

    @property
    def ref_start_sec(self):
        if self.ref_start_end_frame is None or self.fps is None:
            return None
        else:
            ref_start_frame, ref_end_frame = self.ref_start_end_frame
            fps = self.fps
            return float(ref_start_frame) / fps

    @property
    def dis_start_sec(self):
        if self.dis_start_end_frame is None or self.fps is None:
            return None
        else:
            dis_start_frame, dis_end_frame = self.dis_start_end_frame
            fps = self.fps
            return float(dis_start_frame) / fps

    @property
    def fps(self):
        return self.asset_dict['fps'] if 'fps' in self.asset_dict else None

    # ==== str ====

    @property
    def ref_str(self):
        """
        String representation for reference video.
        :return:
        """
        s = ""

        path = get_file_name_without_extension(self.ref_path)
        s += "{path}".format(path=path)

        if self.ref_width_height:
            w, h = self.ref_width_height
            s += "_{w}x{h}".format(w=w, h=h)

        if self.ref_yuv_type != self.DEFAULT_YUV_TYPE:
            s += "_{}".format(self.ref_yuv_type)

        if self.ref_start_end_frame:
            start, end = self.ref_start_end_frame
            s += "_{start}to{end}".format(start=start, end=end)

        return s

    @property
    def dis_str(self):
        """
        String representation for distorted video.
        :return:
        """
        s = ""

        path = get_file_name_without_extension(self.dis_path)
        s += "{path}".format(path=path)

        if self.dis_width_height:
            w, h = self.dis_width_height
            s += "_{w}x{h}".format(w=w, h=h)

        if self.dis_yuv_type != self.DEFAULT_YUV_TYPE:
            s += "_{}".format(self.dis_yuv_type)

        if self.dis_start_end_frame:
            start, end = self.dis_start_end_frame
            s += "_{start}to{end}".format(start=start, end=end)

        return s

    @property
    def quality_str(self):
        """
        String representation for quality-related information
        :return:
        """
        s = ""

        if self.quality_width_height:
            w, h = self.quality_width_height
            if s != "":
                s += "_"
            s += "{w}x{h}".format(w=w, h=h)

        # if resolutions are consistent, no resampling is taking place, so
        # specificying resampling type should be ignored
        if self.resampling_type != self.DEFAULT_RESAMPLING_TYPE and \
                not (self.ref_width_height == self.quality_width_height
                     and self.dis_width_height == self.quality_width_height):
            if s != "":
                s += "_"
            s += "{}".format(self.resampling_type)

        if self.crop_cmd is not None:
            if s != "":
                s += "_"
            s += "crop{}".format(self.crop_cmd)

        if self.pad_cmd is not None:
            if s != "":
                s += "_"
            s += "pad{}".format(self.pad_cmd)

        return s

    def to_string(self):
        """
        The compact string representation of asset, used by __str__.
        :return:
        """
        s = "{dataset}_{content_id}_{asset_id}_{ref_str}_vs_{dis_str}".\
            format(dataset=self.dataset,
                   content_id=self.content_id,
                   asset_id=self.asset_id,
                   ref_str=self.ref_str,
                   dis_str=self.dis_str)
        quality_str = self.quality_str
        if quality_str:
            s += "_q_{quality_str}".format(quality_str=quality_str)
        return s

    def to_normalized_dict(self):
        """
        Similar to self.__dict__ except for excluding workdir (which is random)
        and dir part of ref_path/dis_path.
        :return:
        """
        d = {}
        for key in self.__dict__:
            if key == 'workdir':
                d[key] = ""
            elif key == 'ref_path' or key == 'dis_path':
                d[key] = get_file_name_with_extension(self.__dict__[key])
            else:
                d[key] = self.__dict__[key]
        return d

    def __str__(self):
        """
        Use str(asset) for compact but unique description of asset, for example
        use in file names
        :return:
        """
        return self.to_string()

    def __repr__(self):
        """
        Use repr(asset) for serialization of asset (to be recovered later on)
        :return:
        """
        return self.to_normalized_repr()

    def to_full_repr(self):
        return get_unique_str_from_recursive_dict(self.__dict__)

    def to_normalized_repr(self):
        return get_unique_str_from_recursive_dict(self.to_normalized_dict())

    def __hash__(self):
        return hash(self.to_normalized_repr())

    def __eq__(self, other):
        return self.to_normalized_repr() == other.to_normalized_repr()

    def __ne__(self, other):
        return not self.__eq__(other)

    # ==== workfile ====

    @property
    def ref_workfile_path(self):
        if self.use_path_as_workpath:
            return self.ref_path
        else:
            return "{workdir}/ref_{str}".format(workdir=self.workdir,str=str(self))

    @property
    def dis_workfile_path(self):
        if self.use_path_as_workpath:
            return self.dis_path
        else:
            return "{workdir}/dis_{str}".format(
                workdir=self.workdir,
                str=str(self))

    # ==== bitrate ====

    @property
    def ref_bitrate_kbps_for_entire_file(self):
        """
        :return: the bitrate in Kbps for the entire reference video file. Must
        make sure ref_duration_sec covers the entire file.
        """
        try:
            return os.path.getsize(self.ref_path) / self.ref_duration_sec * 8.0 / 1000.0
        except:
            return None

    @property
    def dis_bitrate_kbps_for_entire_file(self):
        """
        :return: the bitrate in Kbps for the entire reference video file. Must
        make sure ref_duration_sec covers the entire file.
        """
        try:
            return os.path.getsize(self.dis_path) / self.dis_duration_sec * 8.0 / 1000.0
        except:
            return None

    # ==== yuv format ====

    @property
    def ref_yuv_type(self):
        if 'ref_yuv_type' in self.asset_dict:
            if self.asset_dict['ref_yuv_type'] in self.SUPPORTED_YUV_TYPES:
                return self.asset_dict['ref_yuv_type']
            else:
                assert False, "Unsupported YUV type: {}".format(
                    self.asset_dict['ref_yuv_type'])
        elif 'yuv_type' in self.asset_dict:
            if self.asset_dict['yuv_type'] in self.SUPPORTED_YUV_TYPES:
                return self.asset_dict['yuv_type']
            else:
                assert False, "Unsupported YUV type: {}".format(
                    self.asset_dict['yuv_type'])
        else:
            return self.DEFAULT_YUV_TYPE

    @property
    def dis_yuv_type(self):
        if 'dis_yuv_type' in self.asset_dict:
            if self.asset_dict['dis_yuv_type'] in self.SUPPORTED_YUV_TYPES:
                return self.asset_dict['dis_yuv_type']
            else:
                assert False, "Unsupported YUV type: {}".format(
                    self.asset_dict['dis_yuv_type'])
        elif 'yuv_type' in self.asset_dict:
            if self.asset_dict['yuv_type'] in self.SUPPORTED_YUV_TYPES:
                return self.asset_dict['yuv_type']
            else:
                assert False, "Unsupported YUV type: {}".format(
                    self.asset_dict['yuv_type'])
        else:
            return self.DEFAULT_YUV_TYPE

    @property
    def workfile_yuv_type(self):
        """
        for notyuv assets, we want to allow the decoded yuv format to be set by the user
        this is highly relevant to image decoding, where we would like to select yuv444p
        this property tries to read workfile_yuv_type from asset_dict, if it is there it is set
        else it default to default_yuv_type
        """
        supported_yuv_types = list(set(Asset.SUPPORTED_YUV_TYPES) - {'notyuv'})
        if 'workfile_yuv_type' in self.asset_dict:
            workfile_yuv_type = self.asset_dict['workfile_yuv_type']
            assert workfile_yuv_type in supported_yuv_types, "Workfile YUV format {} is not valid, pick: {}".format(
                workfile_yuv_type, str(supported_yuv_types))
            return workfile_yuv_type
        else:
            return self.DEFAULT_YUV_TYPE

    @property
    @deprecated
    def yuv_type(self):
        """ For backward-compatibility """
        return self.dis_yuv_type

    def clear_up_yuv_type(self):
        if 'yuv_type' in self.asset_dict:
            del self.asset_dict['yuv_type']
        if 'ref_yuv_type' in self.asset_dict:
            del self.asset_dict['ref_yuv_type']
        if 'dis_yuv_type' in self.asset_dict:
            del self.asset_dict['dis_yuv_type']

    @property
    def resampling_type(self):
        if 'resampling_type' in self.asset_dict:
            if self.asset_dict['resampling_type'] in self.SUPPORTED_RESAMPLING_TYPES:
                return self.asset_dict['resampling_type']
            else:
                assert False, "Unsupported resampling type: {}".format(
                    self.asset_dict['resampling_type'])
        else:
            return self.DEFAULT_RESAMPLING_TYPE

    @property
    def use_path_as_workpath(self):
        """
        If True, use ref_path as ref_workfile_path, and dis_path as
        dis_workfile_path.
        """
        if 'use_path_as_workpath' in self.asset_dict:
            if self.asset_dict['use_path_as_workpath'] == 1:
                return True
            elif self.asset_dict['use_path_as_workpath'] == 0:
                return False
            else:
                assert False
        else:
            return False

    @use_path_as_workpath.setter
    def use_path_as_workpath(self, bool_value):
        # cannot just assign True/False for ResultStore reason:
        # df = pd.DataFrame.from_dict(ast.literal_eval(result_file.read()))
        # cannot read true/false
        if bool_value is True:
            self.asset_dict['use_path_as_workpath'] = 1
        else:
            self.asset_dict['use_path_as_workpath'] = 0

    @property
    def crop_cmd(self):
        if 'crop_cmd' in self.asset_dict:
            return self.asset_dict['crop_cmd']
        else:
            return None

    @property
    def pad_cmd(self):
        if 'pad_cmd' in self.asset_dict:
            return self.asset_dict['pad_cmd']
        else:
            return None

class NorefAsset(Asset):
    """
    NorefAsset is similar to Asset except that it does not have a reference
    video path ref_path.
    """

    # ==== constructor ====

    def __init__(self, dataset, content_id, asset_id,
                 dis_path,
                 asset_dict,
                 workdir_root=VmafConfig.workdir_path()):
        """
        :param dataset
        :param content_id: ID of content the asset correspond to within dataset
        :param asset_id: ID of asset
        :param dis_path: path to distorted video
        :param asset_dict: dictionary with additional asset properties
        :param workdir_root:
        :return:
        """
        super(NorefAsset, self).__init__(
            dataset,
            content_id,
            asset_id,
            dis_path, # repeat dis_path for both ref_path and dis_path
            dis_path,
            asset_dict,
            workdir_root
        )

    def copy(self, **kwargs):
        # Override Asset.copy, since NorefAsset has a different constructor
        # signature
        new_asset_dict = copy.deepcopy(self.asset_dict)

        # reset the following argument:
        if 'use_path_as_workpath' in new_asset_dict:
            del new_asset_dict['use_path_as_workpath']

        dataset = kwargs['dataset'] if 'dataset' in kwargs else self.dataset
        content_id = kwargs['content_id'] if 'content_id' in kwargs else self.content_id
        asset_id = kwargs['asset_id'] if 'asset_id' in kwargs else self.asset_id
        dis_path = kwargs['dis_path'] if 'dis_path' in kwargs else self.dis_path
        workdir_root = kwargs['workdir_root'] if 'workdir_root' in kwargs else self.workdir_root

        new_asset = self.__class__(dataset, content_id, asset_id,
                                   dis_path, new_asset_dict,
                                   workdir_root)
        return new_asset

    def copy_as_Asset(self, **kwargs):
        """ similar to Noref.copy, except that the returned object is of
        (super)class Asset. """
        new_asset = self.copy()
        new_asset.__class__ = Asset
        return new_asset.copy(**kwargs)