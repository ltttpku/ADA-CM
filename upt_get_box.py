"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits
from interaction_head import InteractionHead

import sys
from hico_list import hico_verb_object_list
from hico_text_label import hico_text_label
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

import pdb
from CLIP_models_adapter_prior2 import CLIP_ResNet, tokenize
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        clip_head: nn.Module,
        clip_pretrained: str,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None
        
    ) -> None:
        super().__init__()
        print('max instance',max_instances)
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = clip_head

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class

        self.clip_head.init_weights(clip_pretrained)
        # pdb.set_trace()
        # self.verb_texts = torch.cat([tokenize(c[0], context_length=3) for c in hico_verb_object_list])
        # self.object_texts = torch.cat([tokenize(c[1], context_length=4,return_sot=False) for c in hico_verb_object_list])
        self.texts = torch.cat([tokenize(v, context_length=13) for k, v in hico_text_label.items()])
        
        # context_length = 13 - 7
        # self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        # nn.init.trunc_normal_(self.contexts)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        train_clip_label = self.compute_text_embeddings()
        self.visual_projection = nn.Linear(1024, 600)
        self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
        
        self.adapter = MLP(1024, 1024//2, 1024, 2)
        self.alpha1 = nn.Parameter(torch.ones([]) * 0.1)

        # self.adapter_t = MLP(1024, 1024//2, 1024, 2)
        # self.beta = nn.Parameter(torch.ones([]) * 0.1)
        # pdb.set_trace()
        self.dicts = {}
    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        # pdb.set_trace()
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        # pdb.set_trace()
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
        
    def compute_text_embeddings(self):
        
        text_embeddings = self.clip_head.text_encoder(self.texts)
        return text_embeddings

    def compute_roi_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []
        # text_embeddings = self.beta * self.adapter_t(text_embeddings) + (1-self.beta) * text_embeddings
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                # pairwise_tokens_collated.append(torch.zeros(
                #     0, 2 * self.representation_size,
                #     device=device)
                # )
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()

            # extract single roi features
            
            # single_boxes = boxes
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[single_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).mean(3).mean(2)
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]

            pdb.set_trace()
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).mean(3).mean(2)
            # pdb.set_trace()
            # pdb.set_trace()
            union_features_mlp = self.adapter(union_features)
            union_features = self.alpha1 * union_features_mlp + (1 - self.alpha1) * union_features
            # pdb.set_trace()
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            logit_scale = self.logit_scale.exp()
            union_features = union_features / union_features.norm(dim=-1, keepdim=True)

            # text_embeddings_bs = text_embeddings[b_idx] / text_embeddings[b_idx].norm(dim=-1, keepdim=True)
            logits = logit_scale * self.visual_projection(union_features)
            # logits = logit_scale * union_features @ text_embeddings_bs.t()
            all_logits.append(logits)
        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated

    def recover_boxes(self, boxes, size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])

        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        
        # labels[x, targets['labels'][y]] = 1
        labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets):
        
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        # pdb.set_trace()
        logits = torch.cat(logits)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        # pdb.set_trace()
        try:
            n_p = len(torch.nonzero(labels))
        except:
            print(n_p)
        # pdb.set_trace()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
        # loss = binary_focal_loss_with_logits(
        #     logits, labels, reduction='sum',
        #     alpha=self.alpha, gamma=self.gamma
        # )
        # print(loss)
        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            if 'scores_a' in list(res.keys()):
                all_sc, sc, lb, bx = res.values()
            else:
                sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)
            if 'scores_a' in list(res.keys()):
                all_sc = all_sc[keep].view(-1, 81)
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            if 'scores_a' in list(res.keys()):
                region_props.append(dict(
                    boxes=bx[keep].cpu().numpy(),
                    scores=sc[keep].cpu().numpy(),
                    labels=lb[keep].cpu().numpy(),
                    hidden_states=hs[keep].cpu().numpy(),
                    all_scores = all_sc[keep].cpu().numpy()
                ))
            else:
                region_props.append(dict(
                    boxes=bx[keep].cpu().numpy(),
                    scores=sc[keep].cpu().numpy(),
                    labels=lb[keep].cpu().numpy(),
                    hidden_states=hs[keep].cpu().numpy()
                ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes):
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images_orig = [im[0] for im in images]
        images_clip = [im[1] for im in images]
        
        # pdb.set_trace()
        # image_sizes = torch.as_tensor([
        #     im.size()[-2:] for im in images_clip
        # ], device=images_clip[0].device)
        # pdb.set_trace()
        # image_sizes = torch.cat([t['orig_size'] for t in targets],device=images_clip[0].device)
        image_sizes = torch.stack([t['orig_size'] for t in targets])
        
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
            images_clip = nested_tensor_from_tensor_list(images_clip)
        features, pos = self.detector.backbone(images_orig)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        
        hs = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])[0]

        outputs_class = self.detector.class_embed(hs)
        outputs_coord = self.detector.bbox_embed(hs).sigmoid()
        # pdb.set_trace()
        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # results = self.postprocessor(results, image_sizes)
        results = self.postprocessor(results, image_sizes,return_score=True)
        
        region_props = self.prepare_region_proposals(results, hs[-1])
        
        return region_props
        

def build_detector(args, class_corr):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    clip_head = CLIP_ResNet(embed_dim=args.clip_visual_output_dim,
                            image_resolution=args.clip_visual_input_resolution,
                            vision_layers=args.clip_visual_layers,
                            vision_width=args.clip_visual_width,
                            vision_patch_size=args.clip_visual_patch_size,
                            context_length=args.clip_text_context_length,
                            transformer_width=args.clip_text_transformer_width,
                            transformer_heads=args.clip_text_transformer_heads,
                            transformer_layers=args.clip_text_transformer_layers)
    detector = UPT(
        detr, postprocessors['bbox'], clip_head, args.clip_dir,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
    )
    return detector
