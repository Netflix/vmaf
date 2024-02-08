# AOM CTC

[AOM](http://aomedia.org/) has specified  [`vmaf`](../../libvmaf/tools/README.md) to be the standard implementation metrics tool according to the AOM common test conditions (CTC).

## Metrics
Using the versioned `--aom_ctc` preset, the following metrics will be computed and logged. These metrics are specified according to the AOM CTC.

* PSNR
* APSNR
* PSNR-HVS
* SSIM
* MS-SSIM
* CIEDE-2000
* VMAF
* VMAF NEG
* CAMBI

## Usage
Basic usage of the tool is described in the [`vmaf` README](../../libvmaf/tools/README.md). Use the versioned `--aom_ctc` presets to register and configure all metrics according to the AOM CTC. Basic AOM CTC usage is as follows:

```
./build/tools/vmaf  \
    --reference reference.y4m \
    --distorted distorted.y4m \
    --aom_ctc v1.0 \
    --output output.xml
```

There are also a few optional command-line settings you may find useful.
* Use `--threads` to set the thread count to be used for multi-threaded computation. This will decrease the overall latency.
* If you prefer a JSON log over the default XML log, use the `--json` flag.

## Output
`XML` and `JSON` logging formats provide per-frame metrics, pooled metrics, and aggregate metrics.

## AOM CTC Version History
* v1.0: `--aom_ctc v1.0`
  * 2021-01-13
  * Fix for lossless comparisons, dB clipping for PSNR/APSNR/SSIM/MS-SSIM according to the AOM CTC.
  * Release: [libvmaf v2.1.1](https://github.com/Netflix/vmaf/releases/tag/v2.1.1)
  * Precompiled static binaries [here](https://github.com/Netflix/vmaf/releases/tag/v2.1.1)

* v1.0: `--aom_ctc v1.0`
  * 2020-12-22
  * Initial CTC release, `--aom_ctc proposed` deprecated.
  * Release: [libvmaf v2.1.0](https://github.com/Netflix/vmaf/releases/tag/v2.1.0)
  * Precompiled static binaries [here](https://github.com/Netflix/vmaf/releases/tag/v2.1.0)

* v2.0: `--aom_ctc v2.0`
  * Release: [libvmaf v2.2.1](https://github.com/Netflix/vmaf/releases/tag/v2.2.1)
  * Precompiled static binaries [here](https://github.com/Netflix/vmaf/releases/tag/v2.2.1)

* v3.0: `--aom_ctc v3.0`
  * 2022-04-05
  * Add CAMBI
  * Release: [libvmaf v2.3.1](https://github.com/Netflix/vmaf/releases/tag/v2.3.1)
  * Precompiled static binaries [here](https://github.com/Netflix/vmaf/releases/tag/v2.3.1)

* v4.0: `--aom_ctc v4.0`
  * Identical to `v3.0`

* v5.0: `--aom_ctc v5.0`
  * Identical to `v4.0`

* v6.0: `--aom_ctc v6.0`
  * 2023-12-07
  * Support bit depth conversion for Y4M inputs
  * Release: [libvmaf v3.0.0](https://github.com/Netflix/vmaf/releases/tag/v3.0.0)
  * Precompiled static binaries [here](https://github.com/Netflix/vmaf/releases/tag/v3.0.0)