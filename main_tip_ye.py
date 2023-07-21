"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from upt_tip_cache_model_free_ye import build_detector
from utils_tip_cache_and_union_ye import custom_collate, CustomisedDLE, DataFactory
# from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory
import pdb, json
import vcoco_text_label

warnings.filterwarnings("ignore")

def vcoco_class_corr():
    """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
    class_corr = []
    for i, (k, v) in enumerate(vcoco_text_label.vcoco_hoi_text_label.items()):
        class_corr.append([i, k[1], k[0]])
    return class_corr

def vcoco_object_n_verb_to_interaction(num_object_cls, num_action_cls, class_corr):
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for i, j, k in class_corr:
            lut[j, k] = i
        return lut.tolist()

def vcoco_object_to_interaction(num_object_cls, _class_corr):
        """
        class_corr: List[(x["action_id"], x["object_id"], x["id"])]
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(num_object_cls)]
        for corr in _class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

def vcoco_interaction_to_verb(_class_corr):
        """
        interaction to verb

        Returns:
            list[list]
        """ 

        inter_to_verb = []
        for i, corr in enumerate(_class_corr):
            inter_to_verb.append(corr[2])
        return inter_to_verb

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16' 
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    
    if args.backbone == 'resnet101':
        detr_backbone = 'R101-DC5' if args.dilation else 'R101'
    elif args.backbone == 'resnet50':
        detr_backbone = 'R50'
    else: 
        raise NotImplementedError("Backbone should be in [resnet50, resnet101]")
    print('[INFO]: detr backbone:', detr_backbone)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root, clip_model_name=args.clip_model_name, detr_backbone=detr_backbone)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root, clip_model_name=args.clip_model_name, detr_backbone=detr_backbone)
    # trainset[0][1]: dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size', 'labels', 'size', 'filename'])
    # trainset[0][0]: (torch.Size([3, 814, 640]), torch.Size([3, 224, 224]))

    if args.dataset == 'vcoco':
        class_corr = vcoco_class_corr()
        trainset.dataset.class_corr = class_corr
        testset.dataset.class_corr = class_corr
        object_n_verb_to_interaction = vcoco_object_n_verb_to_interaction(num_object_cls=len(trainset.dataset.objects), num_action_cls=len(trainset.dataset.actions), class_corr=class_corr)
        trainset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        testset.dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
        object_to_interaction = vcoco_object_to_interaction(num_object_cls=len(trainset.dataset.objects), _class_corr=class_corr)
        trainset.dataset.object_to_interaction = object_to_interaction
        testset.dataset.object_to_interaction = object_to_interaction
        interaction_to_verb = vcoco_interaction_to_verb(_class_corr=class_corr)
        trainset.dataset.interaction_to_verb = interaction_to_verb
        testset.dataset.interaction_to_verb = interaction_to_verb
    
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )

    args.human_idx = 0
    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_interaction
        args.num_classes = 600
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_classes = 24
    # print("[INFO]: obj2target", object_to_target)
    print('[INFO]: num_classes:', args.num_classes)
    if args.dataset == 'vcoco':
        num_anno = None
    else:
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
    upt = build_detector(args, object_to_target, num_anno)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")

        zero_shot_rare_first = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                70, 416,
                389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                345, 189,
                205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                158, 195,
                238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                216, 597,
                77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                55, 50,
                198, 168, 391, 192, 595, 136, 581]

        ap = engine.test_hico(test_loader, args)
        print(ap.shape)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean()*100:.2f},"
            f" rare: {ap[rare].mean()*100:.2f},"
            f" none-rare: {ap[non_rare].mean()*100:.2f},"
        )
        ap_unseen = []
        ap_seen = []
        for i, value in enumerate(ap):
            if i in zero_shot_rare_first: ap_unseen.append(value)
            else: ap_seen.append(value)
        ap_unseen = torch.as_tensor(ap_unseen).mean()
        ap_seen = torch.as_tensor(ap_seen).mean()
        log_stats= f"The mAP is {ap.mean():.4f}, rare: {ap[rare].mean():.4f}, none-rare: {ap[non_rare].mean():.4f}, unseen: {ap_unseen:.4f}, seen: {ap_seen:.4f}"
        print(log_stats)
        
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.named_parameters():
        if n.startswith('adapter'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    for n, p in upt.clip_model.named_parameters():
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj') : 
            p.requires_grad = True
        else: p.requires_grad = False
    
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
        if p.requires_grad]
    }]
    # print(param_dicts)
    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)

    print('number of params:', n_parameters)
 
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
   
    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--visual_mode', default='vit', type=str)
    # add CLIP vision
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    
    ### ViT-B-16 START
    parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    
    parser.add_argument('--clip_text_context_length_vit', default=13, type=int)

    parser.add_argument('--post_process', default=False, action='store_true')
    parser.add_argument('--num_shot', default=1, type=int)

    parser.add_argument('--use_kmeans', default=False, action='store_true')
    parser.add_argument('--file1', default='union_embeddings_cachemodel_crop_padding_zeros_vitb16.p',type=str)
    parser.add_argument('--logits_type', default='HO+U+T', type=str, choices=['HO+U+T', 'U+T', 'HO+T', 'T', 'HO', 'U', "HO+U"]) # 13 -77 # text_add_visual, visual
    # parser.add_argument('--vis_feature_type', default='hum_obj_uni', type=str, choices=('hum_obj_uni', 'hum_uni', 'hum_obj', 'uni'))
    parser.add_argument('--gamma_HO', type=float, default=0.5)
    parser.add_argument('--gamma_U', type=float, default=0.5)
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--rm_duplicate_feat', action='store_true')
    parser.add_argument('--sample_choice', default='uniform', choices=['uniform', 'origin'])
    parser.add_argument('--dic_key', type=str, default='hoi', choices=['hoi', 'verb', 'object'])
    
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'])

    parser.add_argument('--repeat_factor_sampling', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='apply repeat factor sampling to increase the rate at which tail categories are observed')
    
    ## **************** arguments for deformable detr **************** ##
    parser.add_argument('--d_detr', default=False, type=lambda x: (str(x).lower() == 'true'),)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    ## **************** arguments for deformable detr **************** ##
    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
