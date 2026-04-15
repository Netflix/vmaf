# Dataset manifests

One YAML per supported dataset. Shipped copies are **intentionally empty**:
the repository cannot redistribute Netflix / KoNViD / LIVE-VQC / YouTube-UGC
/ BVI-DVC content or MOS scores under their respective licences. Operators
fetch the dataset into a local cache and regenerate the manifest via:

```bash
vmaf-train manifest-scan --dataset <name> --root $VMAF_DATA_ROOT/<name> \
    [--mos-csv path/to/mos.csv]
```

The scanner walks the root for `.yuv` / `.y4m` / `.mp4` / `.mkv` / `.webm`
files, pins each by SHA-256, and emits:

```yaml
name: <dataset>
license: "<licence string>"
entries:
  - key: src01_hrc00_576x324
    path: src01_hrc00_576x324.yuv
    sha256: 9f4c…
    mos: 76.66890482443686
```

`VMAF_DATA_ROOT` points at the local cache root; individual entries resolve
to `${VMAF_DATA_ROOT}/${path}`. MOS CSV format: header row with `key,mos`
columns; unknown keys are ignored, missing keys get `mos: null`.

Never commit the populated manifest if it contains MOS values you are not
licensed to redistribute.
