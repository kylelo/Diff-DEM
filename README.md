<h1 align="left">Diff-DEM: A Diffusion Probabilistic Approach to Digital Elevation Model Void Filling <a href="https://ieeexplore.ieee.org/document/10535979"><img  src="https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg"></a> </h1> 

## Setup
```console
conda env create -f environment.yml
conda activate Diff-DEM
```

## Download Dataset \& Pretrained Model
Download [here](https://drive.google.com/drive/folders/1RXlo2fl-TzGtA1WH5xE3TbzNWsHINln5?usp=sharing) .

Unzip and place dataset under the `Diff-DEM/dataset` of repo e.g. `Diff-DEM/dataset/norway_dem`

Place Diff-DEM pretrained model at `Diff-DEM/pretrained/760_Network.pth`

> The results of `generative_model`, `spline`, `void_fill` tested on Gavriil's dataset are sourced from [repo](https://github.com/konstantg/dem-fill).

## Training
```console
python run.py -p train -c config/dem_completion.json
```

See training progress
```
tensorboard --logdir experiments/train_dem_completion_XXXXXX_XXXXXX
```

## Inference

```console
python run.py -p test -c config/dem_completion.json \
    --resume ./pretrained/760 \
    --n_timestep 512 \
    --data_root ./dataset/norway_dem/benchmark/benchmark_gt.flist \
    --mask_root ./dataset/norway_dem/benchmark/mask_64-96.flist
```

> Tested on NVIDIA RTX3090. Please adjust `batch_size` in JSON file if out of GPU memory.

## Metric
Evaluate the predicted DEM.
For example:
```console
python data/util/tif_metric.py \
    --gt_tif_dir ./dataset/norway_dem/benchmark/gt \
    --mask_dir ./dataset/norway_dem/benchmark/mask/128-160 \
    --algo_dir ./experiments/Diff-DEM/128-160/results/test/0 \
    --normalize
```
Set `--algo_dir` to the DEMs predicted by model e.g. `experiments/test_dem_completion_XXXXXX_XXXXXX/results/test/0` \
For Diff-DEM generated results, use `--normalize`, otherwise do not use.

## Visualization
We view the uint16 DEMs using [ImageJ](https://imagej.net/ij/download.html)

## Acknowledge
This project is based on the following wonderful implementation of the paper [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826) \
https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models

Also, Gavriil's dataset provided in [dem-fill](https://github.com/konstantg/dem-fill).

Lastly, our complete Norway dataset are curated from [Norwegian Mapping Authority](https://hoydedata.no/LaserInnsyn2/).

## Citing
```
@article{lo2024diff,
  title={Diff-DEM: A Diffusion Probabilistic Approach to Digital Elevation Model Void Filling},
  author={Lo, Kyle Shih-Huang and Peters, J{\"o}rg},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024},
  publisher={IEEE}
}
```
