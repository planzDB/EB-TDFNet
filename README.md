## TDFNet


## Introduction

This is the code for our papers:
[Wang, W., Yu, P., Li, M., Zhong, X., He, Y., Su, H., & Zhou, Y. (2025). TDFNet: twice decoding V-Mamba-CNN Fusion features for building extraction. Geo-Spatial Information Science, 1–20. https://doi.org/10.1080/10095020.2025.2514812](https://doi.org/10.1080/10095020.2025.2514812)

## Install

Open the folder **EB-TDFNet** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## DataPreprocess

We follow the [BuildFormer](https://github.com/WangLibo1995/BuildFormer) to pre-process all the datasets.
And then follow the [DSAT-Net](https://github.com/stdcoutzrh/BuildingExtraction) to further process Massachusetts dataset.
More specifically, we use split_1500_to_512.py to resize images from 1500x1500 to 512x512.

```
python EB-TDFNet/tools/split_1500_to_512.py
```

## Training

```
python train_supervision.py -c ./config/whu/tdfnet.py
```

```
python train_supervision.py -c ./config/inria/tdfnet.py
```

```
python train_supervision.py -c ./config/mass/tdfnet.py
```

## Testing

```
python building_seg_test.py -c ./config/whu/tdfnet.py -o /root/autodl-tmp/whu/result/tdfnet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/inria/tdfnet.py -o /root/autodl-tmp/inria/result/tdfnet --rgb -t 'lr'
```

```
python building_seg_test.py -c ./config/mass/tdfnet.py -o /root/autodl-tmp/Massa_512/result/tdfnet --rgb -t 'lr'
```

## Citation

If you find this project useful in your research, please consider citing our papers：
Wang, W., Yu, P., Li, M., Zhong, X., He, Y., Su, H., & Zhou, Y. (2025). TDFNet: twice decoding V-Mamba-CNN Fusion features for building extraction. Geo-Spatial Information Science, 1–20. https://doi.org/10.1080/10095020.2025.2514812

## Acknowledgement

- [BuildFormer](https://github.com/WangLibo1995/BuildFormer)
- [CLCFormer](https://github.com/long123524/CLCFormer)
- [DSAT-Net](https://github.com/stdcoutzrh/BuildingExtraction)
- [ConvNext](https://github.com/facebookresearch/ConvNeXt)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
