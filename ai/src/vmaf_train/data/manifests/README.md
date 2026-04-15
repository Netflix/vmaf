# Dataset manifests

One YAML per supported dataset. Format:

```yaml
name: nflx
license: "Netflix research"
entries:
  - key: src01_hrc00
    path: nflx/src01_hrc00_576x324.yuv
    sha256: "..."
    mos: 76.66890482443686
```

Never commit the YUVs themselves — only the manifest. `VMAF_DATA_ROOT` points
at the local cache root; individual entries resolve to `${VMAF_DATA_ROOT}/${path}`.
