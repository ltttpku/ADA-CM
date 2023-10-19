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
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing) and the pre-extracted bboxes from [HERE](https://drive.google.com/file/d/1xHGr36idtYSzMYGHKvvxMJyTiaq317Ev/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- ADA-CM
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
|   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |   |- hicodet_train_bbox_R50.p
|   |   |- hicodet_test_bbox_R50.p
|   |- vcoco_pkl_files
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |   |- vcoco_train_bbox_R50.p
|   |   |- vcoco_test_bbox_R50.p
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
python main_tip_ye.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir matlab/TF_vcoco/ --num-workers 4 --cache --post_process --dic_key verb --use_multi_hot --num_shot 8 --logits_type HO+U+T --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
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

### Model Zoo

| Dataset |  Backbone  | mAP | Rare | Non-rare | Weights |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET | ResNet-50+ViT-B  | 33.80 | 31.72 | 34.42 | [weights](https://drive.google.com/file/d/1utTPqQkDIvlNhDzAs8mhoSN7FMQjBToH/view?usp=sharing) |
| HICO-DET |ResNet-50+ViT-L  | 38.40 | 37.52 | 38.66 | [weights](https://drive.google.com/file/d/1JqX61ZSDXmDuLz4DPavK3aa1ISG7W8Dj/view?usp=sharing) |


| Dataset |  Backbone  | Scenario 1 | Scenario 2 | Weights |
| ---- |  ----  | ----  | ----  | ----  |
|V-COCO| ResNet-50+ViT-B  | 56.12 | 61.45 | [weights](https://drive.google.com/file/d/13WiXzP08MKSMD-jZrtIpWcyFa7zYXnRE/view?usp=sharing) |
|V-COCO| ResNet-50+ViT-L  | 58.57 | 63.97 | [weights](https://drive.google.com/file/d/1amqgWOPjC8mlHMrmoZj6YzxCFBPLUeww/view?usp=sharing) |

## Citation
If you find our paper and/or code helpful, please consider citing:
```
@article{ting2023hoi,
 title={Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory},
 author={Ting Lei and Fabian Caba and Qingchao Chen and Hailin Ji and Yuxin Peng and Yang Liu},
 year={2023}
 booktitle={ICCV}
 organization={IEEE}
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) for open-sourcing their code.

