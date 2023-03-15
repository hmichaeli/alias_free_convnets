# [Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations](https://arxiv.org/abs/2303.08085)

Official PyTorch implementation

--- 

## Requirements
We provide installation instructions for ImageNet classification experiments here,
based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/9a7b47bd6a6c156a8018dbd0c3b36303d4e564af)
instructions.

### Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2 tensorboardX six
```

The results in the paper are produced with `torch==1.8.0+cu111 torchvision==0.9.0+cu111 timm==0.3.2`.

### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

---
## Train models

### Original ConvNeXt-Tiny
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_tiny \
--drop_path 0.1 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```

###  ConvNeXt-Tiny Baseline (circular convolutions)

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_afc_tiny \
--blurpool_kwargs "{\"filt_size\": 1, \"scale_l2\":false}" \
--activation gelu \
--normalization_type C \
--drop_path 0.1 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```


###  ConvNeXt-Tiny-AFC

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_afc_tiny \
--drop_path 0.1 \
--blurpool_kwargs "{\"filter_type\": \"ideal\", \"scale_l2\":false}" \
--activation up_poly_per_channel \
--activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true}" \
--model_kwargs "{\"stem_mode\":\"activation_residual\", \"stem_activation\": \"lpf_poly_per_channel\"}" \
--stem_activation_kwargs "{\"in_scale\":7, \"out_scale\":7, \"train_scale\":true, \"cutoff\":0.75}" \
--normalization_type CHW2 \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```

### ConvNeXt-Tiny-APS

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1440 main.py \
--model convnext_aps_tiny \
--drop_path 0.1 \
 --blurpool_kwargs "{\"filt_size\": 1}" \
--activation gelu \
--normalization_type C \
--batch_size 32  --update_freq 16 \
--lr 4e-3 \
--model_ema true --model_ema_eval true \
--data_set IMNET \
--data_path </path/to/imagenet> \
--output_dir </path/to/output/dir> \
 --epochs 300 --warmup_epochs 20 \

```
---

## Checkpoints
 
Trained models can be downloaded from:
https://drive.google.com/drive/folders/1IsqMWL8OVKNDQ7CNaHe8F2ox7GDmwMUs?usp=share_link

---

## Acknowledgement
This repository is built using [Truly shift invariant CNNs](https://github.com/achaman2/truly_shift_invariant_cnns/tree/9c319a2f4734745b1a8f2375981750867db1078a) 
and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/tree/9a7b47bd6a6c156a8018dbd0c3b36303d4e564af) repositories.

* Truly shift invariant CNNs: 
  * https://arxiv.org/abs/2011.14214

* ConvNeXt
    * https://arxiv.org/abs/2201.03545
    * [LICENSE](alias_free_convnets/license/convnext_LICENSE.txt)

[//]: # (    * conda version )
[//]: # (Python 3.8	Miniconda3 Linux 64-bit	98.8 MiB	935d72deb16e42739d69644977290395561b7a6db059b316958d97939e9bdf3d)

---
## Citation
If you find this repository helpful, please consider citing:
```
@misc{https://doi.org/10.48550/arxiv.2303.08085,
  doi = {10.48550/ARXIV.2303.08085},
  url = {https://arxiv.org/abs/2303.08085},
  author = {Michaeli, Hagay and Michaeli, Tomer and Soudry, Daniel},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

---