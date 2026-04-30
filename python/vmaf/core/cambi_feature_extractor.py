from vmaf import ExternalProgramCaller
from vmaf.core.feature_extractor import VmafexecFeatureExtractorMixin, FeatureExtractor
from vmaf.tools.reader import YuvReader


class CambiFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "Cambi_feature"
    # VERSION = "0.4"  # Supporting scaled encodes and minor change to the spatial mask
    # VERSION = "0.5"  # Supporting bitdepth converted encodes
    # VERSION = "0.6"  # Include the feature options as part of the feature name
    # VERSION = "0.7"  # Add visibility luminance threshold
    VERSION = "0.8"    # Avoid upscaling when encoding resolution is larger than input resolution

    ATOM_FEATURES = ['cambi']

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'cambi': 'cambi'
    }

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        quality_width, quality_height = asset.quality_width_height
        assert asset.dis_encode_width_height is not None, \
            'For Cambi, dis_encode_width_height cannot be None. One can specify dis_encode_width_height by adding ' \
            'the following fields to asset_dict: 1) dis_enc_width and dis_enc_height, or 2) dis_width and ' \
            'dis_height, or 3) width and height.'
        encode_width, encode_height = asset.dis_encode_width_height

        assert asset.dis_encode_bitdepth is not None, \
            'For Cambi, dis_encode_bitdepth cannot be None. One can specify dis_encode_bitdepth by adding ' \
            'dis_enc_bitdepth field to asset_dict. The supported values are 8, 10, 12, or 16.'
        encode_bitdepth = asset.dis_encode_bitdepth

        if encode_bitdepth > 8 and asset.dis_yuv_type == 'notyuv' and \
                asset.workfile_yuv_type in YuvReader.SUPPORTED_YUV_8BIT_TYPES:
            if encode_bitdepth == 10:
                supported_yuv_types = YuvReader.SUPPORTED_YUV_10BIT_LE_TYPES
            elif encode_bitdepth == 12:
                supported_yuv_types = YuvReader.SUPPORTED_YUV_12BIT_LE_TYPES
            elif encode_bitdepth == 16:
                supported_yuv_types = YuvReader.SUPPORTED_YUV_16BIT_LE_TYPES
            else:
                assert False, "Unsupported encoding bit depth. The supported values are 8, 10, 12, or 16."
            assert False, f"workfile_yuv_type is set to {asset.workfile_yuv_type} for a {encode_bitdepth} bit encode. " \
                          f"This would lead to converting the encode to 8 bit prior to calculating CAMBI and " \
                          f"producing inaccurate results. To compute Cambi in {encode_bitdepth} bit, one can add " \
                          f"workfile_yuv_type field to asset_dict. The supported values are {supported_yuv_types}."

        additional_params = {'enc_bitdepth': encode_bitdepth}
        if encode_width != quality_width or encode_height != quality_height:
            additional_params['enc_width'] = encode_width
            additional_params['enc_height'] = encode_height

        log_file_path = self._get_log_file_path(asset)

        yuv_type = self._get_workfile_yuv_type(asset)
        ref_path = asset.ref_procfile_path
        dis_path = asset.dis_procfile_path
        logger = self.logger

        optional_dict = self.optional_dict if self.optional_dict is not None else dict()
        optional_dict2 = self.optional_dict2 if self.optional_dict2 is not None else dict()

        ExternalProgramCaller.call_vmafexec_single_feature(
            'cambi', yuv_type, ref_path, dis_path, quality_width, quality_height,
            log_file_path, logger, options={**optional_dict, **optional_dict2, **additional_params})


class CambiFullReferenceFeatureExtractor(CambiFeatureExtractor):

    TYPE = "Cambi_FR_feature"
    VERSION = CambiFeatureExtractor.VERSION

    ATOM_FEATURES = ['cambi', 'cambi_full_reference', 'cambi_source']

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'cambi': 'cambi',
        'cambi_full_reference': 'cambi_full_reference',
        'cambi_source': 'cambi_source',
    }

    def _generate_result(self, asset):
        if self.optional_dict is None:
            self.optional_dict = {}

        self.optional_dict["full_ref"] = True

        return super()._generate_result(asset)
