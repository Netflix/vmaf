from vmaf import ExternalProgramCaller
from vmaf.core.feature_extractor import VmafexecFeatureExtractorMixin, FeatureExtractor


class FloatMotionFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_motion_feature"
    # VERSION = "1.0"
    VERSION = "1.1"  # add debug features

    ATOM_FEATURES = ['motion2',
                     'motion',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'motion2': 'motion2',
        'motion': 'motion',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'float_motion', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class IntegerMotionFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "integer_motion_feature"
    # VERSION = "1.0"
    # VERSION = "1.1"  # vectorization
    VERSION = "1.2"  # add debug features

    ATOM_FEATURES = ['motion2',
                     'motion',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'motion2': 'integer_motion2',
        'motion': 'integer_motion',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'motion', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger,
            options={**optional_dict, **optional_dict2})


class FloatVifFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_VIF_feature"
    # VERSION = "1.0"
    VERSION = "1.1"  # add debug features

    ATOM_FEATURES = [
                     'vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                     'vif', 'vif_num', 'vif_den',
                     'vif_num_scale0',
                     'vif_den_scale0',
                     'vif_num_scale1',
                     'vif_den_scale1',
                     'vif_num_scale2',
                     'vif_den_scale2',
                     'vif_num_scale3',
                     'vif_den_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'vif_scale0': 'vif_scale0',
        'vif_scale1': 'vif_scale1',
        'vif_scale2': 'vif_scale2',
        'vif_scale3': 'vif_scale3',
        'vif': 'vif',
        'vif_num': 'vif_num',
        'vif_den': 'vif_den',
        'vif_num_scale0': 'vif_num_scale0',
        'vif_den_scale0': 'vif_den_scale0',
        'vif_num_scale1': 'vif_num_scale1',
        'vif_den_scale1': 'vif_den_scale1',
        'vif_num_scale2': 'vif_num_scale2',
        'vif_den_scale2': 'vif_den_scale2',
        'vif_num_scale3': 'vif_num_scale3',
        'vif_den_scale3': 'vif_den_scale3',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'float_vif', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class IntegerVifFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "integer_VIF_feature"
    # VERSION = "1.0"
    # VERSION = "1.1b"  # vif_enhn_gain_limit with matching_matlab code
    # VERSION = "1.1c"  # update boundary calculation
    # VERSION = "1.1d"  # update to use log2f to replace log2f_approx
    # VERSION = "1.2"  # fix vectorization corner cases
    VERSION = "1.3"  # add debug features

    ATOM_FEATURES = [
                     'vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                     'vif', 'vif_num', 'vif_den',
                     'vif_num_scale0',
                     'vif_den_scale0',
                     'vif_num_scale1',
                     'vif_den_scale1',
                     'vif_num_scale2',
                     'vif_den_scale2',
                     'vif_num_scale3',
                     'vif_den_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'vif_scale0': 'integer_vif_scale0',
        'vif_scale1': 'integer_vif_scale1',
        'vif_scale2': 'integer_vif_scale2',
        'vif_scale3': 'integer_vif_scale3',
        'vif': 'integer_vif',
        'vif_num': 'integer_vif_num',
        'vif_den': 'integer_vif_den',
        'vif_num_scale0': 'integer_vif_num_scale0',
        'vif_den_scale0': 'integer_vif_den_scale0',
        'vif_num_scale1': 'integer_vif_num_scale1',
        'vif_den_scale1': 'integer_vif_den_scale1',
        'vif_num_scale2': 'integer_vif_num_scale2',
        'vif_den_scale2': 'integer_vif_den_scale2',
        'vif_num_scale3': 'integer_vif_num_scale3',
        'vif_den_scale3': 'integer_vif_den_scale3',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'vif', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class FloatAdmFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_ADM_feature"
    # VERSION = "1.0"
    VERSION = "1.1"  # add debug features

    ATOM_FEATURES = ['adm2',
                     'adm_scale0',
                     'adm_scale1',
                     'adm_scale2',
                     'adm_scale3',
                     'adm',
                     'adm_num',
                     'adm_den',
                     'adm_num_scale0',
                     'adm_den_scale0',
                     'adm_num_scale1',
                     'adm_den_scale1',
                     'adm_num_scale2',
                     'adm_den_scale2',
                     'adm_num_scale3',
                     'adm_den_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'adm2': 'adm2',
        'adm_scale0': 'adm_scale0',
        'adm_scale1': 'adm_scale1',
        'adm_scale2': 'adm_scale2',
        'adm_scale3': 'adm_scale3',
        'adm': 'adm',
        'adm_num': 'adm_num',
        'adm_den': 'adm_den',
        'adm_num_scale0': 'adm_num_scale0',
        'adm_den_scale0': 'adm_den_scale0',
        'adm_num_scale1': 'adm_num_scale1',
        'adm_den_scale1': 'adm_den_scale1',
        'adm_num_scale2': 'adm_num_scale2',
        'adm_den_scale2': 'adm_den_scale2',
        'adm_num_scale3': 'adm_num_scale3',
        'adm_den_scale3': 'adm_den_scale3',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'float_adm', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class IntegerPsnrFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = 'integer_PSNR_feature'
    VERSION = "1.0"

    ATOM_FEATURES = ['psnr_y', 'psnr_cb', 'psnr_cr']

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'psnr_y': 'psnr_y',
        'psnr_cb': 'psnr_cb',
        'psnr_cr': 'psnr_cr',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'psnr', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class IntegerAdmFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "integer_ADM_feature"
    # VERSION = "1.0"
    # VERSION = "1.1"  # vectorization; small numerical diff introduced by adm_enhn_gain_limit
    VERSION = "1.2"  # add debug features

    ATOM_FEATURES = ['adm2',
                     'adm_scale0',
                     'adm_scale1',
                     'adm_scale2',
                     'adm_scale3',
                     'adm',
                     'adm_num',
                     'adm_den',
                     'adm_num_scale0',
                     'adm_den_scale0',
                     'adm_num_scale1',
                     'adm_den_scale1',
                     'adm_num_scale2',
                     'adm_den_scale2',
                     'adm_num_scale3',
                     'adm_den_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'adm2': 'integer_adm2',
        'adm_scale0': 'integer_adm_scale0',
        'adm_scale1': 'integer_adm_scale1',
        'adm_scale2': 'integer_adm_scale2',
        'adm_scale3': 'integer_adm_scale3',
        'adm': 'integer_adm',
        'adm_num': 'integer_adm_num',
        'adm_den': 'integer_adm_den',
        'adm_num_scale0': 'integer_adm_num_scale0',
        'adm_den_scale0': 'integer_adm_den_scale0',
        'adm_num_scale1': 'integer_adm_num_scale1',
        'adm_den_scale1': 'integer_adm_den_scale1',
        'adm_num_scale2': 'integer_adm_num_scale2',
        'adm_den_scale2': 'integer_adm_den_scale2',
        'adm_num_scale3': 'integer_adm_num_scale3',
        'adm_den_scale3': 'integer_adm_den_scale3',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'adm', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})


class CIEDE2000FeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = 'CIEDE2000_feature'
    VERSION = "1.0"

    ATOM_FEATURES = ['ciede2000']

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'ciede2000': 'ciede2000',
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        log_file_path = self._get_log_file_path(asset)

        yuv_type=self._get_workfile_yuv_type(asset)
        ref_path=asset.ref_procfile_path
        dis_path=asset.dis_procfile_path
        w=quality_width
        h=quality_height
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'ciede', yuv_type, ref_path, dis_path, w, h, log_file_path, logger,
            options={**optional_dict, **optional_dict2})
