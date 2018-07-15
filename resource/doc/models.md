Models
===================

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

### Invoking Prediction Confidence Interval

As June 2018, we have introduced a way to quantify the level of confidence in VMAF predictions. Each VMAF prediction score can now come with a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. The CI is a consequence of the fact that the VMAF model is trained on a sample of subjective scores, while the population is unknown. The CI is established through bootstrapping on the prediction residue using the full training data. Essentially, it trains multiple models, using "resampling with replacement", on the residue of prediction. Each of the models will introduce a slightly different prediction. The variability of these predictions quantifies the level of confidence -- the more close these predictions, the more confident the prediction using the full data.

To enable CI, use the option `--ci` in the command line tools with a bootstrapping model such as `model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl`.

For example, running

```
./run_vmaf yuv420p 576 324 python/test/resource/yuv/src01_hrc00_576x324.yuv \
python/test/resource/yuv/src01_hrc01_576x324.yuv \
--model model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl --out-fmt json --ci
```

yields:

```
...
    "aggregate": {
        "BOOTSTRAP_VMAF_bagging_score": 73.09994670135325, 
        "BOOTSTRAP_VMAF_score": 75.44304862545658, 
        "BOOTSTRAP_VMAF_stddev_score": 1.2301198524660464, 
        "VMAF_feature_adm2_score": 0.9345878077620574, 
        "VMAF_feature_motion2_score": 3.8953518541666665, 
        "VMAF_feature_vif_scale0_score": 0.36342081156994926, 
        "VMAF_feature_vif_scale1_score": 0.7666473878461729, 
        "VMAF_feature_vif_scale2_score": 0.8628533892781629, 
        "VMAF_feature_vif_scale3_score": 0.9159718691393048, 
        "method": "mean"
    }
}
```

Here, `BOOTSTRAP_VMAF_score` is the final prediction result, similar to `VMAF_score` without the `--ci` option. `BOOTSTRAP_VMAF_stddev_score` is the standard deviation of bootstrapping predictions. If assuming a normal distribution, the 95% CI is `BOOTSTRAP_VMAF_score +/- 1.96 * BOOTSTRAP_VMAF_stddev_score`.
