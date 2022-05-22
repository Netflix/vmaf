# External Resources

This page contains external resources, including a list of 3rd-party tools that are built on top of VMAF. You can also find links to webpages where you can download FFmpeg binaries that support libvmaf.

## Installation Guides
- [How to VMAF (with ffmpeg), journey to the center of despair](https://blog.otterbro.com/how-to-vmaf-ffmpeg/) -- VMAF installation guide on Windows
- [VMAF in FFmpeg – Installation and Usage Guide for Ubuntu](https://ottverse.com/vmaf-ffmpeg-ubuntu-compilation-installation-usage-guide/) -- VMAF installation guide on Ubuntu

## GUI Visualization Tools

- [VideoBench](https://github.com/JNoDuq/videobench) -- VMAF, PSNR and bitrate analyzer
- [FFMetrics](https://github.com/fifonik/FFMetrics) -- Windows-centric GUI for PSNR, SSIM and VMAF visualization
- [NotEnoughQuality](https://github.com/Alkl58/NotEnoughQuality) -- A small GUI handler for VMAF

## FFmpeg-based Tools

- [ffmpeg-quality-metrics](https://github.com/slhck/ffmpeg-quality-metrics) -- command-line tool to calculate PSNR, SSIM and VMAF with FFmpeg
- [EasyVMAF](https://github.com/gdavila/easyVmaf) -- command-line tool with video preprocessing for VMAF inputs
- [Bash wrapper script](https://gist.github.com/Audition-CSBlock/bef34e553132efad883c0f128c46d638) for running `libvmaf` through FFmpeg
- [Video Quality Metrics](https://github.com/BassThatHertz/video-quality-metrics) -- command-line tool which encodes a video using specified x264/x265/AV1 CRF values (or x264/x265 presets) and creates a [table](https://github.com/CrypticSignal/video-quality-metrics#example-table) showing the PSNR/SSIM/VMAF of each encode. In addition, graphs (saved as PNG files) are created where PSNR/SSIM/VMAF score is plotted against frame number. [Here's](https://github.com/CrypticSignal/video-quality-metrics/blob/master/CRF%2023.png) an example

## FFmpeg binaries that support libvmaf
If you do not wish to compile FFmpeg yourself, you can download an FFmpeg binary that supports libvmaf.

- Windows: https://www.gyan.dev/ffmpeg/builds/. Download one of the git builds. The "git-essentials" build will suffice.
- macOS: https://evermeet.cx/ffmpeg/. You should download the **snapshot** build rather than the release build as the latter (at the time of writing) uses v1.5.2 of vmaf.
- Linux (kernel 3.2.0+): https://johnvansickle.com/ffmpeg/. Download the **git** build. Installation instructions, as well as how to add FFmpeg and FFprobe to your PATH, can be found [here](https://www.johnvansickle.com/ffmpeg/faq/).
