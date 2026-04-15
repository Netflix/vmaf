# ffmpeg-patches/

Local patches against FFmpeg for integrating this VMAF fork into the
`libavfilter/vf_libvmaf*` filters.

## Status

Scaffolding only. No patches have been cut yet — upstream FFmpeg still
links against Netflix/libvmaf and the surface additions in this fork
(`--precision`, the SYCL backend, the tiny-AI path) are not yet plumbed
through the filter's option parser.

## Layout

```
ffmpeg-patches/
  README.md          # this file
  series.txt         # quilt-style ordered patch list
  0001-*.patch       # individual patches (not yet written)
```

## How to apply

```bash
cd /path/to/ffmpeg
for p in $(cat /path/to/vmaf/ffmpeg-patches/series.txt); do
    patch -p1 < /path/to/vmaf/ffmpeg-patches/$p
done
```

Or via the helper skill: `/ffmpeg-apply-patches /path/to/ffmpeg`.

## How to regenerate

After editing FFmpeg locally, run `/ffmpeg-build-patches` to diff against
the tracked upstream base and rewrite the numbered patches in place.

## License

BSD-3-Clause-Plus-Patent for patches authored in this repo; patches are
applied against FFmpeg (LGPL / GPL) — the resulting linked binary's
distribution terms are governed by FFmpeg's license, not this one.
