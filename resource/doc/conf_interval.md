# VMAF Confidence Interval

Since v1.3.7 (June 2018), we have introduced a way to quantify the level of confidence that a VMAF prediction entails. With this method, each VMAF prediction score can be accompanied by a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. 

The CI is a consequence of the fact that the VMAF model is trained on a sample of subjective scores, while the population is unknown. The CI is established through [bootstrapping](https://www.jstor.org/stable/2241979) using the full training data. Essentially, the bootstrapping approach trains multiple models. Each of the models will introduce a slightly different prediction. The variability of these predictions quantifies the level of confidence -- the more close these predictions, the more confident the prediction using the full data. More details can be found in [this](presentations/VQEG_SAM_2018_023_VMAF_Variability.pdf) slide deck.

## Implementation Details of Bootstrapping

There are two ways to perform bootstrapping on VMAF. The first one is called plain/vanilla bootstrapping (b) and the latter one is called residue bootstrapping (rb). In the first case, the training data is resampled with replacement to create multiple models and, in the second case, the bootstrapping is performed on the prediction residue. For example, `vmaf_float_b_v0.6.3.json` and `vmaf_rb_v0.6.3.json` are the VMAF models using plain and residue bootstrapping respectively. We recommend using plain bootstrapping, e.g., `vmaf_float_b_v0.6.3.json`. While plain bootstrapping tends to produce larger measurement uncertainty compared to its residue counterpart, it is unbiased with respect to the full VMAF model (which uses the full training data).

## Run in Command Line

To enable CI, use the option `--ci` in the command line tools with a bootstrapping model such as `model/vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.json`. The `--ci` option is available for `run_vmaf`. In [libvmaf](libvmaf/README.md), CI can be enabled by setting the argument `enable_conf_interval` to 1. For the `vmaf` executable, it can automatically detect if a model is a bootstrap model, so just pass in the model path and no `--ci` option is needed.

For example, running

```
./run_vmaf yuv420p 576 324 \
    src01_hrc00_576x324.yuv \
    src01_hrc01_576x324.yuv \
    --model model/vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.pkl \
    --out-fmt json --ci
```

yields:

```
...
    "aggregate": {
        "BOOTSTRAP_VMAF_bagging_score": 74.96366248843681,
        "BOOTSTRAP_VMAF_ci95_high_score": 77.38652840665362,
        "BOOTSTRAP_VMAF_ci95_low_score": 72.98503037587044,
        "BOOTSTRAP_VMAF_score": 75.44304785910772,
        "BOOTSTRAP_VMAF_stddev_score": 1.31289244504376,
        "VMAF_feature_adm2_score": 0.9345878041226809,
        "VMAF_feature_motion2_score": 3.8953518541666665,
        "VMAF_feature_vif_scale0_score": 0.36342081156994926,
        "VMAF_feature_vif_scale1_score": 0.7666473878461729,
        "VMAF_feature_vif_scale2_score": 0.8628533892781629,
        "VMAF_feature_vif_scale3_score": 0.9159718691393048,
        ...
        "method": "mean"
    }
}
```

Here, `BOOTSTRAP_VMAF_score` is the final prediction result and is identical to `VMAF_score` when the `--ci` option is not used. `BOOTSTRAP_VMAF_stddev_score` is the standard deviation of bootstrapping predictions. If assuming a normal distribution, the 95% CI is `BOOTSTRAP_VMAF_score +/- 1.96 * BOOTSTRAP_VMAF_stddev_score`. For a more detailed explanation, please refer to the following section.

## Further Analysis of Bootstrapped Predictions

We assumed, for the sake of simplicity, that the distribution of VMAF predictions is a normal distribution. However, this assumption is not necessarily true. If we do not assume a normal distribution, the 95% CI is defined as `[BOOTSTRAP_VMAF_ci95_low_score, BOOTSTRAP_VMAF_ci95_high_score]`, where `BOOTSTRAP_VMAF_ci95_low_score` and `BOOTSTRAP_VMAF_ci95_high_score` are the 2.5 and 97.5 percentiles respectively. Furthermore, `BOOTSTRAP_VMAF_bagging_score` is the mean of the individual bootstrap models. While `BOOTSTRAP_VMAF_bagging_score` is different from `BOOTSTRAP_VMAF_score`, it is expected that they are relatively similar to each other if the distribution is not very skewed.

## Dataset Validation

CI can also be enabled in [`run_testing`](python.md/#validate-a-dataset) on a dataset. In this case, the `quality_type` must be `BOOTSTRAP_VMAF`, and the `--vmaf-model` must point to the right bootstrapping model. For example:

```
./run_testing \
    BOOTSTRAP_VMAF resource/dataset/NFLX_dataset_public.py \
    --vmaf-model model/vmaf_float_b_v0.6.3/vmaf_float_b_v0.6.3.pkl \
    --cache-result \
    --parallelize
```

Running the command line above will generate scatter plot:

![confidence interval plot](/resource/images/CI.png)

Here each data point (color representing different content) is associated with a 95% CI. It is interesting to note that points on the higher-score end tend to have a tighter CI than points on the lower-score end. This can be explained by the fact that in the dataset to train the VMAF model, there are more dense data points on the higher end than the lower.

## Training Bootstrap Models

To train a bootstrap model, one can use [`run_vmaf_training`](python.md/#train-a-new-model) command line. In the parameter file, the `model_type` must be `BOOTSTRAP_LIBSVMNUSVR`. In `model_param_dict`, one can optionally specify the number of models to be used via `num_models`. See [`vmaf_v6_bootstrap.py`](../../resource/param/vmaf_v6_bootstrap.py) for an example parameter file.

Running the command line below will generate a bootstrap model `test_b_model.pkl`.

```
./run_vmaf_training resource/dataset/NFLX_dataset_public.py \
    resource/param/vmaf_v6_bootstrap.py \
    resource/param/vmaf_v6_bootstrap.py \
    ~/Desktop/test/test_b_model.pkl \
    --cache-result \
    --parallelize
```
