VMAF Confidence Interval
===================

### Invoking Prediction Confidence Interval

Since VDK 1.3.7 (June 2018), we have introduced a way to quantify the level of confidence a VMAF prediction entails. With this method, each VMAF prediction score now come with a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. The CI is a consequence of the fact that the VMAF model is trained on a sample of subjective scores, while the population is unknown. The CI is established through bootstrapping on the prediction residue using the full training data. Essentially, it trains multiple models, using "resampling with replacement", on the residue of prediction. Each of the models will introduce a slightly different prediction. The variability of these predictions quantifies the level of confidence -- the more close these predictions, the more confident the prediction using the full data.

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
