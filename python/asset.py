from common import Parallelizable

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

class Asset(Parallelizable):

    def __init__(self, ref_path, dis_path, asset_dict,
                 workdir_root="../workspace/workdir"):
        """
        :param ref_path: path to reference video
        :param dis_path: path to distorted video
        :param asset_dict: dictionary with additional asset properties
        :param workdir_root:
        :return:
        """
        super(Asset, self).__init__(workdir_root)
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.asset_dict = asset_dict

    @property
    def ref_width_height(self):
        if 'ref_width' in self.asset_dict and 'ref_height' in self.asset_dict:
            return self.asset_dict['ref_width'], self.asset_dict['ref_height']
        elif 'width' in self.asset_dict and 'height' in self.asset_dict:
            return self.asset_dict['width'], self.asset_dict['height']
        else:
            assert False, "reference video's width and height cannot be determined"

    @property
    def dis_width_height(self):
        if 'dis_width' in self.asset_dict and 'dis_height' in self.asset_dict:
            return self.asset_dict['dis_width'], self.asset_dict['dis_height']
        elif 'width' in self.asset_dict and 'height' in self.asset_dict:
            return self.asset_dict['width'], self.asset_dict['height']
        else:
            assert False, "distored video's width and height cannot be determined"

    @property
    def start_end_frame(self):
        """
        :return: asset's start frame and end frame for processing (inclusive)
        """
        if 'start_frame' in self.asset_dict and 'end_frame' in self.asset_dict:
            return self.asset_dict['start_frame'], self.asset_dict['end_frame']
        elif 'start_sec' in self.asset_dict and 'end_sec' in self.asset_dict \
                and 'fps' in self.asset_dict:
            start_frame = int(round(self.asset_dict['start_sec'] * \
                                    self.asset_dict['fps']))
            end_frame = int(round(self.asset_dict['end_sec'] * \
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        elif 'duration_sec' in self.asset_dict and 'fps' in self.asset_dict:
            start_frame = 0
            end_frame = int(round(self.asset_dict['duration_sec'] * \
                                  self.asset_dict['fps'])) - 1
            return start_frame, end_frame
        else:
            # signaling entire video
            return None, None
