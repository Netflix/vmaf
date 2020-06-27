from vmaf import ExternalProgramCaller
from vmaf.core.feature_extractor import VmafrcFeatureExtractorMixin, FeatureExtractor


class FloatMotionFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_motion_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['motion2']

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'motion2': 'motion2',
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

        ExternalProgramCaller.call_vmafrc_single_feature('float_motion', yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


class IntegerMotionFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "integer_motion_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['motion2']

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'motion2': 'integer_motion2',
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

        ExternalProgramCaller.call_vmafrc_single_feature('motion', yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


class FloatVifFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_VIF_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'vif_scale0': 'vif_scale0',
        'vif_scale1': 'vif_scale1',
        'vif_scale2': 'vif_scale2',
        'vif_scale3': 'vif_scale3',
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

        ExternalProgramCaller.call_vmafrc_single_feature('float_vif', yuv_type, ref_path, dis_path, w, h,
                                                         log_file_path, logger, options=self.optional_dict)


class IntegerVifFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "integer_VIF_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                     ]

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'vif_scale0': 'integer_vif_scale0',
        'vif_scale1': 'integer_vif_scale1',
        'vif_scale2': 'integer_vif_scale2',
        'vif_scale3': 'integer_vif_scale3',
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

        ExternalProgramCaller.call_vmafrc_single_feature('vif', yuv_type, ref_path, dis_path, w, h, log_file_path, logger)


class FloatAdmFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = "float_ADM_feature"
    VERSION = "1.0"

    ATOM_FEATURES = ['adm2',
                     ]

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
        'adm2': 'adm2',
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

        ExternalProgramCaller.call_vmafrc_single_feature('float_adm', yuv_type, ref_path, dis_path, w, h,
                                                         log_file_path, logger, options=self.optional_dict)


class IntegerPsnrFeatureExtractor(VmafrcFeatureExtractorMixin, FeatureExtractor):

    TYPE = 'integer_PSNR_feature'
    VERSION = "1.0"

    ATOM_FEATURES = ['psnr_y', 'psnr_cb', 'psnr_cr']

    ATOM_FEATURES_TO_VMAFRC_KEY_DICT = {
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

        ExternalProgramCaller.call_vmafrc_single_feature('psnr', yuv_type, ref_path, dis_path, w, h, log_file_path, logger)
