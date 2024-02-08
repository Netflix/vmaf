# Frequently Asked Questions

#### Q: When computing VMAF on low-resolution videos (480-pixel height, for example), why the scores look so high, even when there are visible artifacts?

A: It is associated with the underlying assumption of VMAF on the subject viewing distance and display size.

Fundamentally, any perceptual quality model should take into account the viewing distance and the display size (or the ratio between the two). The same distorted video, if viewed closed-up, could contain more visual artifacts hence yield lower perceptual quality.

In the case of the default VMAF model (`model/vmaf_float_v0.6.1.pkl`), which is trained to predict the quality of videos displayed on a 1080p HDTV in a living-room-like environment, all the subjective data were collected in such a way that the distorted videos get rescaled to 1080 resolution and displayed with a viewing distance of three times the screen height (3H), or angular resolution of 60 pixels/degree. Effectively, what the VMAF model trying to capture is the perceptual quality of a 1080 video displayed from 3H away. That’s the implicit assumption of the default VMAF model.

Now, think about what it means when the VMAF score is calculated on a reference/distorted video pair of 480 resolution. It is similar to as if the 480 video is CROPPED from a 1080 video. If the height of the 480 video is H’, then H’ = 480 / 1080 * H = 0.44 * H, where H is the height of the 1080 video displayed. As a result, VMAF is modeling the viewing distance of 3*H = 6.75*H’. In other words, if you calculate VMAF on the 480 resolution video pair, you are going to predict the perceptual quality of viewing distance 6.75 times its height. This is going to hide a lot of artifacts, hence yielding a very high score. 

One implication of the observation above is that, one should NOT compare the absolute VMAF score of a 1080 video with the score of a 480 video obtained at its native resolution -- it is an apples-to-oranges comparison.

If, say, for a distorted video of 480 resolution, we still want to predict its quality viewing from 3 times the height (not 6.75), how can this be achieved? 

- If the 480 distorted video comes with a source (reference) video of 1080 resolution, then the right way to do it is to upsample the 480 video to 1080, and calculate the VMAF at 1080, together with its 1080 source.

- If the 480 distorted video has only a 480 reference, then you can still upsample both distorted/reference to 1080, and calculate VMAF. A caveat is, since the VMAF model was not trained with upsampled references, the prediction would not be as accurate as 1).

#### Q: Why the included SSIM tool produces numerical results off compared to other tools?

A: The SSIM implementation in the VMAF package includes an empirical downsampling process, as described at the Suggested Usage section of [SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/). The other implementations, such as the SSIM filter in FFmpeg, does not include this step.

#### Q: Why the aggregate VMAF score sometimes may bias "easy" content too much? [Issue #20](https://github.com/Netflix/vmaf/issues/20)

A: By default, the VMAF output reports the aggregate score as the average (i.e. arithmetic mean) of the per-frame scores mostly for its simplicity, as well as for consistency with other metrics (e.g. mean PSNR). There are psycho-visual evidences, however, suggest that human opinions tend to weigh more heavily towards the worst-quality frames. It is an open question what the optimal way to pool the per-frame scores is, as it also depends on many factors, such as the time scale of the pooling (seconds vs minutes).

To provide some flexibility, in command-line tools *run_vmaf*, *run_psnr*, *run_vmaf_in_batch*, *run_vmaf_training* and *run_testing*, among others, we added a hidden option `--pool pool_method`, where `pool_method` is among `mean`, `harmonic_mean`, `median`, `min`, `perc5`, `perc10` and `perc20` (percx means x-percentile).

#### Q: Will VMAF work on 4K videos?

A: The default VMAF model at `model/vmaf_v0.6.1.pkl` was trained on videos encoded at resolutions *up to* 1080p. It is still useful for measuring 4K videos, if you are interested in a relative score. In other words, for two 4K videos A and B with A perceptually better than B, the VMAF scores will tell you so too. However, if you are interested in an absolute score, say if a 4K video is perceptually acceptable, you may not get an accurate answer.

As of VDK v1.3.7 (June 2018), we have added a new 4K model at `model/vmaf_4k_v0.6.1.json`, which is trained to predict 4KTV viewing at distance of 1.5X the display height. Refer to [this](resource/doc/models.md/#predict-quality-on-a-4ktv-screen-at-15h) section for details.

#### Q: Will VMAF work on applications other than HTTP adaptive streaming?

A: VMAF was designed with HTTP adaptive streaming in mind. Correspondingly, in terms of the types of video artifacts, it only considers compression artifact and scaling artifact (read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post for more details). The perceptual quality of other artifacts (for example, artifacts due to packet losses or transmission errors) MAY be predicted inaccurately.

#### Q: Can I pass encoded H264/VP9/H265 bitstreams to VMAF as input? [Issue #55](https://github.com/Netflix/vmaf/issues/55)

A: Yes, you can. The paved path forward is to use FFmpeg's `libvmaf` filter option. Refer to the [FFmpeg documentation](resource/doc/ffmpeg.md) for details.

#### Q: When I compare a video with itself as reference, I expect to get a perfect score of VMAF 100, but what I see is a score like 98.7. Is there a bug?

A: VMAF does not guarantee that you get a perfect score in this case, but you should get a score close enough. Similar things would happen to other machine learning-based predictors (another example is VQM-VFD).

#### Q: How is the VMAF package versioned?

A: Since the package has been growing and there were confusion on what this VMAF number should be in the VERSION file, it is decided to stick to the convention that this VMAF version should only be related to the version of the default model for the `VmafQualityRunner`. Whenever there is a numerical change to the VMAF result in running the default model, this number is going to be updated. For anything else, we are going to use the VDK version number. For `libvmaf`, whenever there is an interface change or numerical change to the VMAF result, the version number at `https://github.com/Netflix/vmaf/blob/master/src/libvmaf/libvmaf.pc` is going to be updated to the latest VDK number.

#### Q: If I train a model using the `run_vmaf_training` process with some dataset, and then I run the `run_testing` process with that trained model and the same dataset, why wouldn't I get the same results (SRCC, PCC, and RMSE)? [Issue #191](https://github.com/Netflix/vmaf/issues/191)

A: This is due to the slightly different workflows used by `run_vmaf_training` and `run_testing`. In `run_vmaf_training`, the feature scores (elementary metric scores) from each frame are first extracted,  each feature is then temporally pooled (by arithmetic mean) to form a feature score per clip. The per-clip feature scores are then fit with the subjective scores to obtain the trained model. The reported SRCC, PCC and RMSE are the fitting result. In `run_testing`, the per-frame feature scores are first extracted, then the prediction model is applied on a per-frame basis, resulting "per-frame VMAF score". The final score for the clip is arithmetic mean of the per-frame scores. As you can see, there is a re-ordering of the 'temporal pooling' and 'prediction' operators. If the features from a clip are constant, the re-ordering will not have an impact. In practice, we find the numeric difference to be small.

### Q: How do I use VMAF with downscaled videos?

A: If you have a distorted video that was scaled down (e.g. for adaptive streaming) and want to calculate VMAF, you can use FFmpeg with `libvmaf` to perform the re-scaling for you.

For example, to upscale the distorted video to 1080p:

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex "[0:v]scale=1920x1080:flags=bicubic[main];[main][1:v]libvmaf" -f null -
```

This scales the first input video (`0:v`) and forwards it to VMAF (`libvmaf`) with the label `main`, where it is compared against the second input video, `1:v`. More details can be found [here](resource/doc/ffmpeg.md).

### Q: Why are the PSNR values capped at 60 dB for 8-bit inputs and 72 dB for 10-bit inputs in the package's implementation?

A: The peak PSNR values follow a rule-of-thumb formula of `6 * N + 12`, where `N` is the bit depth. In the most recent support of 12-bit and 16-bit inputs, we have followed this convention and capped the PSNR at 84 dB and 108 dB, respectively.

This formula above is an approximation (for mnemonics purpose) of a more precise formula `10 * log10( (2^N - 1)^2 / (1 / 12))`, where `2^N - 1` is the peak signal of an N-bit representation, and `1 / 12` is the variance of a `Uniform[0, 1]`-distributed quantization noise, as a result of rounding a signal to its integer representation. For example, with `N = 8`, the formula yields 58.92 dB. The variance of the quantization noise captures the fundamental limit of an N-bit representation.

One may argue that even with 8-bit inputs, the PSNR could go to infinity if the distorted signal matches the reference signal exactly. Notice that in this case, it is because that the reference signal itself happens to be in 8-bit representation; in general, one could easily imagine that the reference signal comes from a high-bit or floating-point representation, and the PSNR is subject to the fundamental limit characterized by the formula above. Thus, we think that imposing a bit depth-based cap on the PSNR sets a more realistic goal on the signal fidelity one tries to achieve.
