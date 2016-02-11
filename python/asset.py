__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os

from common import Parallelizable
from tools import get_file_name_without_extension, get_file_name_with_extension, \
    get_unique_str_from_recursive_dict
import config

class Asset(Parallelizable):

    def __init__(self, dataset, content_id, asset_id,
                 ref_path, dis_path,
                 asset_dict,
                 workdir_root= config.ROOT + "/workspace/workdir"):
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
        super(Asset, self).__init__(workdir_root)
        self.dataset = dataset
        self.content_id = content_id
        self.asset_id = asset_id
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.asset_dict = asset_dict

    @staticmethod
    def from_dict(asset_dict):
        return {}

    # ==== width and height ====

    @property
    def ref_width_height(self):
        """
        :return: width and height of reference video. If None, it signals that
        width and height should be figured out in other means (e.g. FFMPEG).
        """
        if 'ref_width' in self.asset_dict \
                and 'ref_height' in self.asset_dict:
            return self.asset_dict['ref_width'], \
                   self.asset_dict['ref_height']
        elif 'width' in self.asset_dict \
                and 'height' in self.asset_dict:
            return self.asset_dict['width'], \
                   self.asset_dict['height']
        else:
            return None

    @property
    def dis_width_height(self):
        """
        :return: width and height of distorted video. If None, it signals that
        width and height should be figured out in other means (e.g. FFMPEG)
        """
        if 'dis_width' in self.asset_dict \
                and 'dis_height' in self.asset_dict:
            return self.asset_dict['dis_width'], \
                   self.asset_dict['dis_height']
        elif 'width' in self.asset_dict \
                and 'height' in self.asset_dict:
            return self.asset_dict['width'], \
                   self.asset_dict['height']
        else:
            return None

    @property
    def quality_width_height(self):
        """
        :return: width and height at which the quality is measured at. either
        'quality_width' and 'quality_height' have to present in asset_dict,
        or ref and dis's width and height must be equal, which will be used
        as the default quality width and height.
        """
        assert ('quality_width' in self.asset_dict
                and 'quality_height' in self.asset_dict) or \
               (self.ref_width_height == self.dis_width_height)

        if 'quality_width' in self.asset_dict \
                and 'quality_height' in self.asset_dict:
            return self.asset_dict['quality_width'], \
                   self.asset_dict['quality_height']
        else:
            return self.ref_width_height

    # ==== start and end frame ====

    @property
    def ref_start_end_frame(self):
        """
        :return: reference video's start frame and end frame for processing
        (inclusive). If None, it signals that the entire video should be
        processed.
        """
        if 'ref_start_frame' in self.asset_dict \
                and 'ref_end_frame' in self.asset_dict:
            return self.asset_dict['ref_start_frame'], \
                   self.asset_dict['ref_end_frame']
        elif 'start_frame' in self.asset_dict \
                and 'end_frame' in self.asset_dict:
            return self.asset_dict['start_frame'], \
                   self.asset_dict['end_frame']
        elif 'start_sec' in self.asset_dict \
                and 'end_sec' in self.asset_dict \
                and 'fps' in self.asset_dict:
            start_frame = int(round(self.asset_dict['start_sec'] *
                                    self.asset_dict['fps']))
            end_frame = int(round(self.asset_dict['end_sec'] *
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        elif 'duration_sec' in self.asset_dict \
                and 'fps' in self.asset_dict:
            start_frame = 0
            end_frame = int(round(self.asset_dict['duration_sec'] *
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        else:
            return None

    @property
    def dis_start_end_frame(self):
        """
        :return: distorted video's start frame and end frame for processing
        (inclusive). If None, it signals that the entire video should be
        processed.
        """
        if 'dis_start_frame' in self.asset_dict and \
                        'dis_end_frame' in self.asset_dict:
            return self.asset_dict['dis_start_frame'], \
                   self.asset_dict['dis_end_frame']
        elif 'start_frame' in self.asset_dict and \
                        'end_frame' in self.asset_dict:
            return self.asset_dict['start_frame'], \
                   self.asset_dict['end_frame']
        elif 'start_sec' in self.asset_dict \
                and 'end_sec' in self.asset_dict \
                and 'fps' in self.asset_dict:
            start_frame = int(round(self.asset_dict['start_sec'] *
                                    self.asset_dict['fps']))
            end_frame = int(round(self.asset_dict['end_sec'] *
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        elif 'duration_sec' in self.asset_dict \
                and 'fps' in self.asset_dict:
            start_frame = 0
            end_frame = int(round(self.asset_dict['duration_sec'] *
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        else:
            return None

    # ==== duration ====

    @property
    def ref_duration_sec(self):
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

    # ==== str ====

    @property
    def ref_str(self):
        s = ""

        path = get_file_name_without_extension(self.ref_path)
        s += "{path}".format(path=path)

        if self.ref_width_height:
            w, h = self.ref_width_height
            s += "_{w}x{h}".format(w=w, h=h)

        if self.ref_start_end_frame:
            start, end = self.ref_start_end_frame
            s += "_{start}to{end}".format(start=start, end=end)

        return s

    @property
    def dis_str(self):
        s = ""

        path = get_file_name_without_extension(self.dis_path)
        s += "{path}".format(path=path)

        w, h = self.dis_width_height
        s += "_{w}x{h}".format(w=w, h=h)

        if self.dis_start_end_frame:
            start, end = self.dis_start_end_frame
            s += "_{start}to{end}".format(start=start, end=end)

        return s

    @property
    def quality_str(self):
        s = ""

        if self.quality_width_height:
            w, h = self.quality_width_height
            if s != "":
                s += "_"
            s += "{w}x{h}".format(w=w, h=h)

        return s

    def to_string(self):
        """
        The unique string representation of asset, used by both __str__ and
        __repr__.
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
        return get_unique_str_from_recursive_dict(self.to_normalized_dict())

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def from_repr(rp):
        """
        Reconstruct Asset from repr string.
        :return:
        """
        d = eval(rp)
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

    # ==== workfile ====

    @property
    def ref_workfile_path(self):
        return "{workdir}/ref_{str}".format(
            workdir=self.workdir,
            str=str(self))

    @property
    def dis_workfile_path(self):
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
            return os.path.getsize(self.ref_path) / \
                   self.ref_duration_sec * 8.0 / 1000.0
        except:
            return None

    @property
    def dis_bitrate_kbps_for_entire_file(self):
        """
        :return: the bitrate in Kbps for the entire reference video file. Must
        make sure ref_duration_sec covers the entire file.
        """
        try:
            return os.path.getsize(self.dis_path) \
                   / self.dis_duration_sec * 8.0 / 1000.0
        except:
            return None

    @property
    def yuv_type(self):
        """
        Assuming ref/dis files are both YUV and the same type, return the type
        (yuv420p, yuv422p, yuv444p)
        :return:
        """
        if 'yuv_type' in self.asset_dict:
            if self.asset_dict['yuv_type'] in ['yuv420p', 'yuv422p', 'yuv444p']:
                return self.asset_dict['yuv_type']
            else:
                assert False, "Unknown YUV type: {}".format(
                    self.asset_dict['yuv_type'])
        else:
            return 'yuv420p'
