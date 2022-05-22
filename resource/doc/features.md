# Features

Features are individual inputs that (VMAF) models may use to fuse into a final quality metric score. Note that an existing quality metric can serve as a feature of a new quality metric. Features can also be enabled individually without fusing (e.g., when using VMAF via FFmpeg)."

## Core features

The following core features are part of the pre-trained VMAF models, and they have been introduced in the original [VMAF tech blog article](https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652) from 2016:

### Visual Information Fidelity (VIF)

VIF is a well-adopted image quality metric based on the premise that quality is complementary to the measure of information fidelity loss. In its original form, the VIF score is measured as a loss of fidelity combining four scales. In VMAF, we adopt a modified version of VIF where the loss of fidelity in each scale is included as an elementary metric.

It has been described in Sheikh, Hamid R., and Alan C. Bovik. "Image information and visual quality." IEEE Transactions on image processing 15.2 (2006): 430-444.

### Motion2

This is a simple measure of the temporal difference between adjacent frames. This is accomplished by calculating the average absolute pixel difference for the luminance component.

### ADM

ADM was previously named Detail Loss Metric (DLM), described in S. Li, F. Zhang, L. Ma, and K. Ngan, “Image Quality Assessment by Separately Evaluating Detail Losses and Additive Impairments,” IEEE Transactions on Multimedia, vol. 13, no. 5, pp. 935–949, Oct. 2011.

DLM is an image quality metric based on the rationale of separately measuring the loss of details which affects the content visibility, and the redundant impairment which distracts viewer attention. The original metric combines both DLM and additive impairment measure (AIM) to yield a final score. In VMAF, only the DLM part is added as an elementary metric. Particular care was taken for special cases, such as black frames, where numerical calculations for the original formulation break down.

## Additional features

The following additional features are available and explained on dedicated pages:

- [CAMBI](cambi.md)
- CIEDE2000
- MS-SSIM
- PSNR
- PSNR-HVS
- SSIM
