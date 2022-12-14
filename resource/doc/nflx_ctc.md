# NFLX CTC

## Metrics
Using the versioned `--nflx_ctc` preset, the following metrics will be computed and logged. These metrics are specified according to the NFLX CTC.

* PSNR
* APSNR
* SSIM
* VMAF
* VMAF NEG
* CAMBI

## Usage
Basic usage of the tool is described in the [`vmaf` README](../../libvmaf/tools/README.md). Use the versioned `--nflx_ctc` presets to register and configure all metrics according to the NFLX CTC. Basic usage is as follows:

```
./build/tools/vmaf  \
    --reference reference.y4m \
    --distorted distorted.y4m \
    --nflx_ctc v1.0 \
    --json \
    --output output.json
```

There are also a few optional command-line settings you may find useful.
* Use `--threads` to set the thread count to be used for multi-threaded computation. This will decrease the overall latency.

## Output
`JSON` logging formats provide per-frame metrics, pooled metrics, and aggregate metrics.

## NFLX CTC Version History
* v1.0: `--nflx_ctc v1.0`
  * 2023-01-01
  * Initial CTC release
