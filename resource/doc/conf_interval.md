VMAF Confidence Interval
===================

Since VDK 1.3.7 (June 2018), we have introduced a way to quantify the level of confidence that a VMAF prediction entails. With this method, each VMAF prediction score can be accompanied by a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. 

The CI is a consequence of the fact that the VMAF model is trained on a sample of subjective scores, while the population is unknown. The CI is established through [bootstrapping on the prediction residue](http://www.jstor.org/stable/2241979) using the full training data. Essentially, it trains multiple models, using "resampling with replacement", on the residue of prediction. Each of the models will introduce a slightly different prediction. The variability of these predictions quantifies the level of confidence -- the more close these predictions, the more confident the prediction using the full data. More details can be found in [this](VQEG_SAM_2018_023_VMAF_Variability.pdf) slide deck.

### Run in Command Line

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

### Dataset Validation

CI can also be enabled in [`run_testing`](VMAF_Python_library.md/#validate-a-dataset) on a dataset. In this case, the `quality_type` must be `BOOTSTRAP_VMAF`, and the `--vmaf-model` must point to the right bootstrapping model. For example:

```
./run_testing BOOTSTRAP_VMAF resource/dataset/NFLX_dataset_public.py \
  --vmaf-model model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl --cache-result --parallelize
```

Running the command line above will generate scatter plot:

![confidence interval plot](/resource/images/CI.png)

Here each data point (color representing different content) is associated with a 95% CI. It is interesting to note that points on the higher-score end tend to have a tighter CI than points on the lower-score end. This can be explained by the fact that in the dataset to train the VMAF model, there are more dense data points on the higher end than the lower.

### Training Bootstrap Models

To train a bootstrap model, one can use [`run_vmaf_training`](VMAF_Python_library.md/#train-a-new-model) command line. In the parameter file, the `model_type` must be `RESIDUEBOOTSTRAP_LIBSVMNUSVR`. In `model_param_dict`, one can optionally specify the number of models to be used via `num_models`. See [`vmaf_v6_residue_bootstrap.py`](../../resource/param/vmaf_v6_residue_bootstrap.py) for an example parameter file.

Running the command line below will generate a bootstrap model `test_rb_model.pkl`.

```
./run_vmaf_training resource/dataset/NFLX_dataset_public.py \
  resource/param/vmaf_v6_residue_bootstrap.py \
  resource/param/vmaf_v6_residue_bootstrap.py \
  ~/Desktop/test/test_rb_model.pkl --cache-result --parallelize
```
