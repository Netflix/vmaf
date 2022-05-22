from abc import ABC

from vmaf.core.cambi_feature_extractor import CambiFeatureExtractor, CambiFullReferenceFeatureExtractor
from vmaf.core.quality_runner import QualityRunnerFromFeatureExtractor
from vmaf.tools.decorator import override


class CambiQualityRunner(QualityRunnerFromFeatureExtractor, ABC):

    TYPE = 'Cambi'
    VERSION = "0.4" # Supporting scaled encodes and minor change to the spatial mask

    @override(QualityRunnerFromFeatureExtractor)
    def _get_feature_extractor_class(self):
        return CambiFeatureExtractor

    @override(QualityRunnerFromFeatureExtractor)
    def _get_feature_key_for_score(self):
        return 'cambi'

class CambiFullReferenceQualityRunner(QualityRunnerFromFeatureExtractor, ABC):

    TYPE = 'Cambi_FR'
    VERSION = "0.4" # Supporting scaled encodes and minor change to the spatial mask

    @override(QualityRunnerFromFeatureExtractor)
    def _get_feature_extractor_class(self):
        return CambiFullReferenceFeatureExtractor

    @override(QualityRunnerFromFeatureExtractor)
    def _get_feature_key_for_score(self):
        return 'cambi_full_reference'