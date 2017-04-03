from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import PsnrQualityRunner, VmafQualityRunner, \
    SsimQualityRunner, MsSsimQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.core.vqm_quality_runner import VqmGeneralQualityRunner, \
    VqmVfdQualityRunner

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

if __name__ == '__main__':

    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_x2.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_x2.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width':576, 'height':324,
                              # 'quality_width':640, 'quality_height':480,
                              'fps': 20,
                              })

    result_store = None
    # result_store = FileSystemResultStore()

    runner_classs = [
        PsnrQualityRunner,
        VmafQualityRunner,
        SsimQualityRunner,
        MsSsimQualityRunner,
        VqmGeneralQualityRunner,
        VqmVfdQualityRunner,
    ]

    for runner_class in runner_classs:
        runner = runner_class(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=result_store
        )
        runner.run()

        result = runner.results[0]

        print result
        print '-------------------------'

