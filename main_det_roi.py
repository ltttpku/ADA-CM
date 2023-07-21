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

# from upt_bbox import build_detector
from upt_get_box import build_detector
# from upt_det_roi import build_detector
from utils_det_roi import custom_collate, CustomisedDLE, DataFactory
import pdb
warnings.filterwarnings("ignore")

def generate_class_coor(dataset):
    obj_to_action = list(dataset.object_to_action.values().copy()) ## one-based index to zero-based index
    obj_lst = dataset.objects.copy()[1:] # one-based index to zero-based index
    action_lst = dataset.actions.copy()
    interaction_lst = []
    for o_idx, ac_lst in enumerate(obj_to_action):
        for a in ac_lst:
            assert a in action_lst
            interaction_lst.append((a, obj_lst[o_idx]))
    # pdb.set_trace()
    class_coor = [] #Class correspondence matrix in zero-based index [ [hoi_idx, obj_idx, verb_idx], ... ]
    for i, interaction in enumerate(interaction_lst):
        class_coor.append([i, obj_lst.index(interaction[-1]), action_lst.index(interaction[0])])
    
    def get_object_n_verb_to_interaction(class_coor, num_object_cls, num_action_cls) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([num_object_cls, num_action_cls], None)
        for i, j, k in class_coor:
            lut[j, k] = i
        return lut.tolist()
    
    object_n_verb_to_interaction = get_object_n_verb_to_interaction(class_coor, len(obj_lst), len(action_lst))

    def get_object_to_interaction(class_coor, num_object_cls) -> List[list]:
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(num_object_cls)]
        for corr in class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    object_to_interaction = get_object_to_interaction(class_coor, len(obj_lst))

    def get_interaction_to_verb(class_coor) -> List[list]:
        """
        interaction to verb

        Returns:
            list[list]
        """ 

        inter_to_verb = []
        for i, corr in enumerate(class_corr):
            inter_to_verb.append(corr[2])
        return inter_to_verb
    
    interaction_to_verb = get_interaction_to_verb(class_coor)


    dataset.class_coor = class_coor
    dataset.object_n_verb_to_interaction = object_n_verb_to_interaction
    dataset.object_to_interaction = object_to_interaction
    dataset.interaction_to_verb = interaction_to_verb


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

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)
    # if args.dataset == 'vcoco':
    #     generate_class_coor(trainset.dataset)

    # pdb.set_trace()
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    # test_loader = DataLoader(
    #     dataset=testset,
    #     collate_fn=custom_collate, batch_size=1,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=False,
    #     sampler=torch.utils.data.SequentialSampler(testset)
    # )
    print('[INFO]: using training set in test loader!!!!')
    test_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(trainset)
    ) 

    args.human_idx = 0
    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_classes = 117
        # object_to_target = train_loader.dataset.dataset.object_to_interaction
        # args.num_classes = 600
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_classes = 24
    
    upt = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")
    # pdb.set_trace()
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
        # if args.dataset == 'vcoco':
        #     raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        if args.backbone == 'resnet101':
            detr_backbone = 'R101-DC5' if args.dilation else 'R101'
        elif args.backbone == 'resnet50':
            detr_backbone = 'R50'
        else:
            raise NotImplementedError("Backbone should be in [resnet50, resnet101]")

        ap = engine.test_hico(test_loader, args.dataset, detr_backbone=detr_backbone)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean():.4f},"
            f" rare: {ap[rare].mean():.4f},"
            f" none-rare: {ap[non_rare].mean():.4f}," 
        )
        print(args.resume)
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.clip_head.named_parameters():
        
        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj') or 'adaptermlp' in n: 
            p.requires_grad = True
            print(n)
        else: p.requires_grad = False
    # param_dicts = [{
    #     "params": [p for n, p in upt.named_parameters()
    #     if "interaction_head" in n and p.requires_grad]
    # }]
    # param_dicts = [{
    #     "params": [p for n, p in upt.named_parameters()
    #     if p.requires_grad or n.startswith('clip_head.visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj')]
    # }]
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
        if p.requires_grad]
    }]
    # print(param_dicts)
    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)
    # print()

    print('number of params:', n_parameters)
    # pdb.set_trace()
    # if os.path.exists(args.resume):
    #     print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     # upt.load_state_dict(checkpoint['model_state_dict'])
    #     optim = checkpoint['optim_state_dict']

    # else:
    #     print(f"=> Rank {rank}: start from a randomly initialised model")
        
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
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-drop', default=20, type=int)
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
    # add CLIP model resenet 
    parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    parser.add_argument('--clip_visual_width', default=64, type=int)
    parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    parser.add_argument('--clip_text_context_length', default=13, type=int)

    # add CLIP vision
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)
    # parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-32.pt', type=str)
    parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=672, type=int)
    parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=32, type=int)
    parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    parser.add_argument('--clip_text_context_length_vit', default=13, type=int)

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
