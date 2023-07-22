# [ICCV 2023] Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory

## Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- ADA-CM
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Our code is built upon [CLIP](https://github.com/openai/CLIP). Install the local package of CLIP:
```
cd CLIP && python setup.py develop && cd ..
```

3. Download the CLIP weights to `checkpoints/pretrained_clip`.
```
|- ADA-CM
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
|   |       |- ViT-L-14-336px.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |
| V-COCO | [weights](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing) |


```
|- ADA-CM
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
|   |   |- detr-r50-vcoco.pth
:   :   :
```

## Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- ADA-CM
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
|   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |- vcoco_pkl_files
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
:   :      
```

## TrainingFree Mode
### HICO-DET
```
python main_tip_ye.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/test --eval --post_process --use_multi_hot --logits_type HO+U+T --num_shot 8 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt
```
### V-COCO
Cache detection results for evaluation on V-COCO:
```
python main_tip_ye.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir matlab/TF_vcoco/ --num-workers 4 --cache --post_process --dic_key verb --use_multi_hot --num_shot 8 --logits_type HO+U+T  --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16_TF.p 
```

For V-COCO, we did not implement evaluation utilities, and instead use the utilities provided by [Gupta et al.](https://github.com/ywchao/ho-rcnn). Refer to these [instructions](https://github.com/fredzzhang/upt/discussions/14) for more details.


## FineTuning Mode
### HICO-DET
#### Train on HICO-DET:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt 
```

#### Test on HICO-DET:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --eval --resume CKPT_PATH
```

### V-COCO
#### Training on V-COCO
```
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco-injector-r50 --use_insadapter --num_classes 24 --use_multi_hot --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p  --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt
```

#### Cache detection results for evaluation on V-COCO
```
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco-injector-r50 --use_insadapter --num_classes 24 --use_multi_hot --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p  --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --cache --resume CKPT_PATH
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) for open-sourcing their code.