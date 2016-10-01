# Frequently Asked Questions

**Q: When computing VMAF on low-resolution videos (480 height, for example), why the score look so high, even when there are visible artifacts?**

A: It is associated with the underlying assumption of VMAF on the subject viewing distance and display size.

Fundamentally, any perceptual quality model should take the viewing distance and the display size (or the ratio between the two) as input. The same distorted video, if viewed closed-up, could contain more visual artifacts hence yield lower perceptual quality. Think about SSIM: at [SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/), in section “Suggested Usage”, the authors make the recommendation of downsampling an image to 256 height (or width) before calculating SSIM. Effectively, this is to normalize all images to a fixed setting of viewing distance / display size ratio such that it could correlate better with subjective data.

In the case of VMAF, all the subjective data were collected in such a way that the distorted videos get rescaled to 1080 resolution and displayed with a viewing distance of three times the screen height (3H). Effectively, what the VMAF model trying to capture is the perceptual quality of a 1080 video displayed from 3H away. That’s the implicit assumption of our current VMAF model.

Now, think about what it means when the VMAF score is calculated on a reference/distorted video pair of 480 resolution. It is similar to as if the 480 video is CROPPED from a 1080 video. If the height of the 480 video is H’, then H’ = 480 / 1080 * H = 0.44 * H, where H is the height of the 1080 video displayed. As a result, VMAF is modeling the viewing distance of 3*H = 6.75*H’. In other words, if you calculate VMAF on the 480 resolution video pair, you are going to predict the perceptual quality of viewing distance 6.75 times its height. This is going to hide a lot of artifacts, hence yielding a very high score.

If, say, for a distorted video of 480 resolution, we still want to predict its quality viewing from 3 times the height (not 6.75), how can this be achieved? 

- If the 480 distorted video comes with a source (reference) video of 1080 resolution, then the right way to do it is to upsample the 480 video to 1080, and calculate the VMAF at 1080, together with its 1080 source.

- If the 480 distorted video has only a 480 reference, then you can still upsample both distorted/reference to 1080, and calculate VMAF. A caveat is, since the VMAF model was not trained with upsampled references, the prediction would not be as accurate as 1). In the future, we do plan to release a new VMAF version which also takes into account the reference video’s resolution.

**Q: Why the included SSIM tool produces numerical results off compared to other tools?**

A: The ssim implementation in the VMAF package includes an empirical downsampling process, as described at the Suggested Usage section of [SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/). The other implementations, such as the SSIM filter in FFmpeg, does not include this step.
