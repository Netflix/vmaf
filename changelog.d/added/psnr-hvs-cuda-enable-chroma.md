Add `enable_chroma` option (default `false`) to `psnr_hvs_cuda` feature
extractor, mirroring the ADR-0453 pattern from `integer_psnr_cuda`. When
`false`, only the luma plane is processed and `psnr_hvs` reports luma dB.
YUV400P sources always produce luma-only output regardless of the option.
