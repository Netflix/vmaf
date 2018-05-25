# Frequently Asked Questions

**Q: When computing VMAF on low-resolution videos (480 height, for example), why the scores look so high, even when there are visible artifacts?**

A: It is associated with the underlying assumption of VMAF on the subject viewing distance and display size.

Fundamentally, any perceptual quality model should take the viewing distance and the display size (or the ratio between the two) as input. The same distorted video, if viewed closed-up, could contain more visual artifacts hence yield lower perceptual quality.

In the case of VMAF, all the subjective data were collected in such a way that the distorted videos get rescaled to 1080 resolution and displayed with a viewing distance of three times the screen height (3H). Effectively, what the VMAF model trying to capture is the perceptual quality of a 1080 video displayed from 3H away. That’s the implicit assumption of our current VMAF model.

Now, think about what it means when the VMAF score is calculated on a reference/distorted video pair of 480 resolution. It is similar to as if the 480 video is CROPPED from a 1080 video. If the height of the 480 video is H’, then H’ = 480 / 1080 * H = 0.44 * H, where H is the height of the 1080 video displayed. As a result, VMAF is modeling the viewing distance of 3*H = 6.75*H’. In other words, if you calculate VMAF on the 480 resolution video pair, you are going to predict the perceptual quality of viewing distance 6.75 times its height. This is going to hide a lot of artifacts, hence yielding a very high score.

If, say, for a distorted video of 480 resolution, we still want to predict its quality viewing from 3 times the height (not 6.75), how can this be achieved? 

- If the 480 distorted video comes with a source (reference) video of 1080 resolution, then the right way to do it is to upsample the 480 video to 1080, and calculate the VMAF at 1080, together with its 1080 source.

- If the 480 distorted video has only a 480 reference, then you can still upsample both distorted/reference to 1080, and calculate VMAF. A caveat is, since the VMAF model was not trained with upsampled references, the prediction would not be as accurate as 1). In the future, we do plan to release a new VMAF version which also takes into account the reference video’s resolution.

**Q: Why the included SSIM tool produces numerical results off compared to other tools?**

A: The SSIM implementation in the VMAF package includes an empirical downsampling process, as described at the Suggested Usage section of [SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/). The other implementations, such as the SSIM filter in FFmpeg, does not include this step.

**Q: Why the aggregate VMAF score sometimes may bias "easy" content too much? [Issue #20](https://github.com/Netflix/vmaf/issues/20)**

A: By default, the VMAF output reports the aggregate score as the average (i.e. mean) per-frame score mostly for its simplicity, as well as for consistency with other metrics (e.g. mean PSNR). There are psycho-visual evidences, however, suggest that human opinions tend to weigh more heavily towards the worst-quality frames. It is an open question what the optimal way to pool the per-frame scores is, as it also depends on many factors, such as the time scale of the pooling (seconds vs minutes).

To provide some flexibility, in CLIs *run_vmaf*, *run_psnr*, *run_vmaf_in_batch*, *run_vmaf_training* and *run_testing*, we added a hidden option *--pool pool_method*, where *pool_method* is among *mean*, *harmonic_mean*, *median*, *min*, *perc5*, *perc10* and *perc20* (percx means x-percentile).

**Q: Will VMAF work on 4K videos?**

A: The current VMAF model (v0.6.1) was trained on videos encoded at *up to* 1080p resolution. It is still useful for measuring 4K videos, if you are interested in a relative score. In other words, for two 4K videos A and B with A perceptually better than B, the VMAF scores will tell you so too. However, if you are interested in an absolute score, say if a 4K video is perceptually acceptable, you may not get an accurate answer.

The future plan is to publish a model specifically trained on 4K videos.

**Q: Will VMAF work on applications other than HTTP adaptive streaming?**

A: VMAF was designed with HTTP adaptive streaming in mind. Correspondingly, in terms of the types of video artifacts, it only considers compression artifact and scaling artifact (read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post for more details). The perceptual quality of other artifacts (for example, artifacts due to packet losses or transmission errors) may be predicted inaccurately.

**Q: Can I pass encoded H264/VP9/H265 to VMAF as input? [Issue #55](https://github.com/Netflix/vmaf/issues/55)**

A: Yes, you can. You can transcode an encoded video to raw YUV stream (e.g. by FFmpeg) and pipe it to VMAF. An example can be found [here](https://github.com/Netflix/vmaf/blob/master/ffmpeg2vmaf).

**Q: When I compare a video with itself as reference, I expexct to get a perfect score of VMAF 100, but what I see is a score like 98.7. Is there a bug?**

A: VMAF doesn't guarantee that you get a perfect score in this case, but you should get a score close enough. Similar things would happen to other machine learning-based predictors (another example is VQM-VFD).
