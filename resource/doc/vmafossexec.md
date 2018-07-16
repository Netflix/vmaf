`vmafossexec` -- a C++ "Wrapper" Executable
===================

The VDK package combines feature extraction implementation in C and the rest scripting code in Python. The Python layer allows fast prototyping, but sometimes deploying the Python dependency in production is a pain. Under [`wrapper`](../../wrapper), we provide a C++ implementation `vmafossexec` that has no dependency on Python.

To build `vmafossexec`, make the `ptools` library first, followed by the executable itself:

```
cd ptools; make; cd ../wrapper; make; cd ..;
```

Under root, run `vmafossexec` as:

```
wrapper/vmafossexec yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  model/vmaf_v0.6.1.pkl \
  --log vmaf_output.xml \
  --psnr --ssim --ms-ssim \
  --thread 0 --subsample 5
```

For VMAF v0.6.1, the model file is `model/vmaf_v0.6.1.pkl`.

The options `--psnr`, `--ssim` and `--ms-ssim` also allow reporting PSNR, SSIM and MS-SSIM results, respectively. The option `--thread` specifies the number of threads to use. Apply `--thread 0` to use all threads available. The option `--subsample` specifies the subsampling of frames to speed up calculation. For example, `--subsample 5` calculates VMAF on one of every 5 frames. The following plot shows the trend of how the subsample number impacts the processing speed (based on the [Netflix Public Dataset](datasets.md#netflix-public-dataset) of 1080p videos, with PSNR, SSIM and MS-SSIM calculation enabled):

![subsample](/resource/images/subsample.png)

Optionally, one can test `vmafossexec` by running the [`vmafossexec_test.py`](../../python/test/vmafossexec_test.py) script (this requires Python 2 and the `numpy`, `scipy`, `pandas` and `sklearn` packages):

```
pip install --user numpy scipy pandas sklearn
PYTHONPATH=python/src python python/test/vmafossexec_test.py
```

Expect all tests pass.
