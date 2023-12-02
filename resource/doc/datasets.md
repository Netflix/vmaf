# Datasets

We also provide two sample datasets including the video files and the properly formatted dataset files in Python. They can be used as sample datasets to train and test custom VMAF models.

## Netflix Public Dataset

We provide a dataset publicly available to the community for training, testing and verification of results purposes. The dataset file is at [`NFLX_dataset_public.py`](../dataset/NFLX_dataset_public.py), and the videos (in YUV420P format) can be downloaded [here](https://drive.google.com/folderview?id=0B3YWNICYMBIweGdJbERlUG9zc0k&usp=sharing) (please request for access and we will grant it). Each file name is in the format of `{content name}_{expert score}_{height in pixels}_{bitrate in Kbps}.yuv`. For example, `BirdsInCage_85_720_1050.yuv` is an decompressed video from content `BirdInCage`, with an expert opinion score 85 out of 100, compressed with 720p resolution at 1050 Kbps. Note that the expert opinion score is different from the MOS score offered in `NFLX_dataset_public.py`, which are from a panel of non-expert subjects.

## VQEG HD3 Dataset

We also provide an example dataset file containing video file names from VQEG (Video Quality Expert Group) HD3 videos. The dataset file is at [`VQEGHD3_dataset.py`](../dataset/VQEGHD3_dataset.py), and the videos are available for downloading from [http://www.cdvl.org/](http://www.cdvl.org/). After login, choose 'Advanced Search', select 'VQEG Subjective Tests' in the dataset dropdown, and search for the keyword 'vqeghd3'. This will return a single combined ZIP file of ~84 GB, 'VQEG HDTV Test, vqeghd3' as the individual sequences are no longer available for download. The dataset file includes from `src01` to `src09` except for `src04`, which overlaps with the Netflix Public Dataset, and `hrc04`, `hrc07`, `hrc16`, `hrc17`, `hrc18`, `hrc19`, `hrc20` and `hrc21`, which are the most relevant distortion types to adaptive streaming. After downloading the videos, convert them to YUV420P format.
