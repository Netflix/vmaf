Models
===================

### Predict Quality on a 1080p screen at 3H

In the case of the default VMAF model (`model/vmaf_v0.6.1.pkl`), which is trained to predict the quality of videos displayed on a 1080p HDTV in a living-room-like environment, all the subjective data were collected in such a way that the distorted videos get rescaled to 1080 resolution and displayed with a viewing distance of three times the screen height (3H). Effectively, what the VMAF model trying to capture is the perceptual quality of a 1080 video displayed from 3H away. That’s the implicit assumption of our default VMAF model.

### Predict Quality on a Cellular Phone Screen

VMAF v0.6.1 and later support a custom quality model for cellular phone screen viewing. This model can be invoked by adding `--phone-model` option in the commands `run_vmaf`, `run_vmaf_in_batch` (but also in `run_testing` and `vmafossexec` which are introduced the following sections):

```
./run_vmaf yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --phone-model

./run_vmaf_in_batch resource/example/example_batch_input --parallelize \
  --phone-model
```

This model is trained using subjective data collected in a lab experiment, based on the [absolute categorical rating (ACR)](https://en.wikipedia.org/wiki/Absolute_Category_Rating) methodology, with the exception that after viewing a video sequence, a subject votes on a continuous scale (from "bad" to "excellent"), instead of the more conventional five-level discrete scale. The test content are video clips selected from the Netflix catalog, each 10 seconds long. For each clip, a combination of 6 resolutions and 3 encoding parameters are used to generate the processed video sequences, resulting 18 impairment conditions for testing. Instead of fixating the viewing distance, each subject is instructed to view the video at a distance he/she feels comfortable with. In the trained model, the score ranges from 0 to 100, which is linear with the subjective voting scale, where roughly "bad" is mapped to score 20, and "excellent" is mapped to score 100.

Invoking the phone model will generate VMAF scores higher than in the regular model, which is more suitable for laptop, TV, etc. viewing conditions. An example VMAF–bitrate relationship for the two models is shown below:

![regular vs phone model](/resource/images/phone_model.png)

From the figure it can be interpreted that due to the factors of screen size and viewing distance, the same distorted video would be perceived as having a higher quality when viewed on a phone screen than on a laptop/TV screen, and when the quality score reaches its maximum (100), further increasing the encoding bitrate would not result in any perceptual improvement in quality.

### Predict Quality on a 4KTV Screen at 1.5H

As June 2018, we have added a new 4K VMAF model at `model/vmaf_4k_v0.6.1.pkl`, which predicts the subjective quality of video displayed on a 4KTV and viewed from the distance of 1.5 times the height of the display device (1.5H). This model is trained with subjective data collected in a lab experiment, using the ACR methodology. The viewing distance of 1.5H is the critical distance for a human subject to appreciate the quality of 4K content (see [recommendation](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2022-0-201208-I!!PDF-E.pdf)).

To invoke this model, specify the model path using the `--model` option. For example:

```
./run_vmaf yuv420p 3840 2160 ref_path dis_path --model model/vmaf_4k_v0.6.1.pkl
```
