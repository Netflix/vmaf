from vmaf import ExternalProgramCaller
from vmaf.core.feature_extractor import VmafexecFeatureExtractorMixin, FeatureExtractor


class PsnrhvsFeatureExtractor(VmafexecFeatureExtractorMixin, FeatureExtractor):

    TYPE = 'PSNRHVS_feature'
    VERSION = "1.0"

    ATOM_FEATURES = ['psnr_hvs', 'psnr_hvs_y', 'psnr_hvs_cb', 'psnr_hvs_cr']

    ATOM_FEATURES_TO_VMAFEXEC_KEY_DICT = {
        'psnr_hvs': 'psnr_hvs',
        'psnr_hvs_y': 'psnr_hvs_y',
        'psnr_hvs_cb': 'psnr_hvs_cb',
        'psnr_hvs_cr': 'psnr_hvs_cr',
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
            'psnr_hvs', yuv_type, ref_path, dis_path, w, h,
            log_file_path, logger, options={**optional_dict, **optional_dict2})
