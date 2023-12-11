from vmaf import ExternalProgramCaller
from vmaf.core.feature_extractor import VmafexecFeatureExtractorMixin, FeatureExtractor


class CambiFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = "Cambi_feature"
    # VERSION = "0.4" # Supporting scaled encodes and minor change to the spatial mask
    VERSION = "0.5"  # Supporting bitdepth converted encodes

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