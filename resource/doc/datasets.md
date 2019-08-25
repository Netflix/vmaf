Datasets
===================

We also provide two sample datasets including the video files and the properly formatted dataset files in Python. They can be used as sample datasets to train and test custom VMAF models.

### Netflix Public Dataset

We provide a dataset publicly available to the community for training, testing and verification of results purposes. The dataset file is at [`NFLX_dataset_public.py`](../dataset/NFLX_dataset_public.py), and the videos (in YUV420P format) can be downloaded [here](https://drive.google.com/folderview?id=0B3YWNICYMBIweGdJbERlUG9zc0k&usp=sharing).

### VQEG HD3 Dataset

We also provide an example dataset file containing video file names from VQEG (Video Quality Expert Group) HD3 videos. The dataset file is at [`VQEGHD3_dataset.py`](../dataset/VQEGHD3_dataset.py), and the videos is available for downloading from [http://www.cdvl.org/](http://www.cdvl.org/). After login, choose menu 'find videos', and search use keyword 'vqeghd3'. The dataset file includes from `src01` to `src09` except for `src04`, which overlaps with the Netflix Public Dataset, and `hrc04`, `hrc07`, `hrc16`, `hrc17`, `hrc18`, `hrc19`, `hrc20` and `hrc21`, which are the most relevant distortion types to adaptive streaming. After downloading the videos, convert them to YUV420P format.

