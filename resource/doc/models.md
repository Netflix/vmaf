# Models

This repository offers a number of pre-trained VMAF models to be used in different scenarios. Besides the default VMAF model which predicts the quality of a video displayed on a HDTV in a living-room viewing condition, this repository also includes a number of additional models, covering mobile phone and 4KTV viewing conditions.

## Predict Quality on a 1080p HDTV screen at 3H

The default VMAF model (`model/vmaf_v0.6.1.json`) is trained to predict the quality of videos displayed on a 1080p HDTV in a living-room-like environment. All the subjective data were collected in such a way that the distorted videos (with native resolutions of 1080p, 720p, 480p etc.) get rescaled to 1080 resolution and shown on the 1080p display with a viewing distance of three times the screen height (3H). Note that 3H is the critical distance for a viewer to appreciate 1080p resolution sharpness (see [recommendation](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2022-0-201208-I!!PDF-E.pdf)).

This model is trained using subjective data collected in a lab experiment, based on the [absolute categorical rating (ACR)](https://en.wikipedia.org/wiki/Absolute_Category_Rating) methodology, with the exception that after viewing a video sequence, a subject votes on a continuous scale (from "bad" to "excellent", with evenly spaced markers of "poor", "fair" and "good" in between), instead of the more conventional five-level discrete scale. The test content are video clips selected from the Netflix catalog, each 10 seconds long. For each clip, a combination of 6 resolutions and 3 encoding parameters are used to generate the processed video sequences, resulting 18 impairment conditions for testing. 

The raw subjective scores collected are then cleaned up using the MLE methodology, as described in [SUREAL](https://github.com/Netflix/sureal). The aggregate subjective scores after clean-up are mapped to a score in the VMAF score, where "bad" is mapped to roughly score 20, and "excellent" to 100.

## Predict Quality on a Cellular Phone Screen

The default VMAF model (`model/vmaf_v0.6.1.json`) also offers a custom model for cellular phone screen viewing. This model can be invoked by adding `--phone-model` option in the commands `run_vmaf`, `run_vmaf_in_batch`, and `run_testing` e.g.:

```shell script
./run_vmaf yuv420p 576 324 \
    src01_hrc00_576x324.yuv \
    src01_hrc01_576x324.yuv \
    --phone-model
```

The subjective experiment uses similar video sequences as the default 1080p HDTV model, except that they were watched on a cellular phone screen (Samsung S5 with resolution 1920x1080). Instead of fixating the viewing distance, each subject is instructed to view the video at a distance he/she feels comfortable with. In the trained model, the score ranges from 0 to 100, which is linear with the subjective voting scale, where roughly "bad" is mapped to score 20, and "excellent" is mapped to score 100.

Invoking the phone model will generate VMAF scores higher than in the regular model, which is more suitable for laptop, TV, etc. viewing conditions. An example VMAF–bitrate relationship for the two models is shown below:

![regular vs phone model](/resource/images/phone_model.png)

From the figure it can be interpreted that due to the factors of screen size and viewing distance, the same distorted video would be perceived as having a higher quality when viewed on a phone screen than on a laptop/TV screen, and when the quality score reaches its maximum (100), further increasing the encoding bitrate would not result in any perceptual improvement in quality.

## Predict Quality on a 4KTV Screen at 1.5H

As of v1.3.7 (June 2018), we have added a new 4K VMAF model at `model/vmaf_4k_v0.6.1.json`, which predicts the subjective quality of video displayed on a 4KTV and viewed from the distance of 1.5 times the height of the display device (1.5H). Again, this model is trained with subjective data collected in a lab experiment, using the ACR methodology (notice that it uses the original 5-level discrete scale instead of the continuous scale). The viewing distance of 1.5H is the critical distance for a human subject to appreciate the quality of 4K content (see [recommendation](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2022-0-201208-I!!PDF-E.pdf)). More details can be found in [this](presentations/VQEG_SAM_2018_025_VMAF_4K.pdf) slide deck.

To invoke this model, specify the model path using the `--model` option. For example:

```shell script
./run_vmaf yuv420p 3840 2160 \
    ref_path \
    dis_path \
    --model model/vmaf_4k_v0.6.1.json
```

## Disabling Enhancement Gain (NEG mode)

For comparing encoders, VMAF offers a special mode, called *No Enhancement Gain*. This is described in the [following blog post](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30):

> One unique feature about VMAF that differentiates it from traditional metrics such as PSNR and SSIM is that VMAF can capture the visual gain from image enhancement operations, which aim to improve the subjective quality perceived by viewers. (…) However, in codec evaluation, it is often desirable to measure the gain achievable from compression without taking into account the gain from image enhancement during pre-processing.

To disable enhancement gain, use the versions of the model files ending with `neg`.

More details on the reasoning behind NEG have been shared in [this tech memo](https://docs.google.com/document/d/1dJczEhXO0MZjBSNyKmd3ARiCTdFVMNPBykH4_HMPoyY/edit#heading=h.oaikhnw46pw5). A high-level overview can be found in [this slide deck](https://docs.google.com/presentation/d/1ZVQPsA4N6K8uGW3aFgw4Ei9w953nYORUUPvgpigOq58/edit?usp=sharing).

## What are the Differences between Individual Models?

There are no material differences comparing 0.6.2 and 0.6.3 to 0.6.1.

0.6.2 and 0.6.3 were retrained later on the same dataset with the same hyperparameters, but based on the elementary features that have been slightly improved.

The models 0.6.2 and 0.6.3 also come with a way to compute the prediction confidence interval based on bootstrapping.
