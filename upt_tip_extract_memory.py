"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from __future__ import annotations
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits
from interaction_head import InteractionHead

import sys
from hico_list import hico_verb_object_list,hico_verbs
from hico_text_label import hico_text_label, hico_obj_text_label
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

import pdb
# from CLIP_models import CLIP_ResNet, tokenize
from CLIP_models_adapter_prior2 import CLIP_ResNet, tokenize
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle
from tools import forward_chunks

_tokenizer = _Tokenizer()
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
    def __init__(self, args,
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
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = clip_head
        self.visual_output_dim = args.clip_visual_output_dim_vit

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class


        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model, _ = clip.load(args.clip_model_name, device=device)

        text_inputs = torch.cat([clip.tokenize(hico_text_label[id]) for id in hico_text_label.keys()])
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs.to(device))
        self.logit_scale = self.clip_model.logit_scale
        self.text_embedding = text_embedding

        self.dicts = {}

    def get_clip_feature(self,image):
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # pdb.set_trace()
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        # x = self.clip_model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        return local_feat
    def _reset_parameters(self):
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def compute_roi_embeddings_targets_invalid_pairs(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict], targets: List[dict]):
        device = features.device
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for i, props in enumerate(region_props):
            local_features = features[i]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            filename = targets[i]['filename']
            verbs = targets[i]['labels']
            hois = targets[i]['hoi']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1: continue

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
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point


            # compute predict bbox with the gts
            
            gt_bx_h = self.recover_boxes(targets[i]['boxes_h'], targets[i]['size'])
            gt_bx_o = self.recover_boxes(targets[i]['boxes_o'], targets[i]['size'])
            # pdb.set_trace()
            x, y = torch.nonzero(torch.min(
                box_iou(sub_boxes, gt_bx_h),
                box_iou(obj_boxes, gt_bx_o)
            ) >= self.fg_iou_thresh).unbind(1)
            verbs = verbs[y]
            hois = hois[y]
            # pdb.set_trace()
            all_pairs = torch.arange(len(sub_boxes))
            # invalid_indexes = torch.as_tensor(list(set(all_pairs.numpy()) - set(x.cpu().numpy())))
            invalid_indexes = torch.as_tensor(list(set(all_pairs.numpy()) & set(x.cpu().numpy())))
            if len(invalid_indexes) ==0 : continue
            # union_boxes = torch.cat([lt,rb],dim=-1)[invalid_indexes]
            # sub_boxes = sub_boxes[invalid_indexes]
            # obj_boxes = obj_boxes[invalid_indexes]
            sub_boxes = sub_boxes[x]
            obj_boxes = obj_boxes[x]
            union_boxes = torch.cat([lt,rb],dim=-1)[x]
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[sub_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[obj_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # pdb.set_trace()
            assert sub_boxes.shape[0] == len(verbs) 
            self.dicts[filename] = dict(boxes_h=sub_boxes.cpu().numpy(),
            boxes_o=obj_boxes.cpu().numpy(),verbs=verbs.cpu().numpy(), hois=hois.cpu().numpy(),
            union_boxes=union_boxes.cpu().numpy(), union_features=union_features.cpu().numpy(),
            huamn_features=huamn_features.cpu().numpy(), object_features=object_features.cpu().numpy(),
            objects=labels[y_keep][invalid_indexes].cpu().numpy()
            )


        return 

    def compute_roi_embeddings_targets(self, features: OrderedDict, image_shapes: Tensor, targets_region_props: List[dict]):
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for i, targets in enumerate(targets_region_props):
            # pdb.set_trace()
            local_features = features[i]
            gt_bx_h = (box_ops.box_cxcywh_to_xyxy(targets['boxes_h']) * scale_fct[i][None,:]).half()
            gt_bx_o = (box_ops.box_cxcywh_to_xyxy(targets['boxes_o']) * scale_fct[i][None,:]).half()
            verbs = targets['labels']
            hois = targets['hoi']
            filename = targets['filename']
            objects_label = targets['object']
            lt = torch.min(gt_bx_h[..., :2], gt_bx_o[..., :2]) # left point
            rb = torch.max(gt_bx_h[..., 2:], gt_bx_o[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1).half()
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            # pdb.set_trace()
            self.dicts[filename] = dict(boxes_h=gt_bx_h.cpu().numpy(),
            boxes_o=gt_bx_o.cpu().numpy(), verbs=verbs.cpu().numpy(), hois=hois.cpu().numpy(),
            union_boxes=union_boxes.cpu().numpy(), union_features=union_features.cpu().numpy(),
            huamn_features=huamn_features.cpu().numpy(), object_features=object_features.cpu().numpy(),
            objects=objects_label.cpu().numpy()
            )
            
        return 
    
    def compute_roi_embeddings_targets_crop(self, features: OrderedDict, image_shapes: Tensor, targets_region_props: List[dict]):

        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for i, targets in enumerate(targets_region_props):
            local_features = features[i][1:]
            global_feature = features[i][0] ## feature of the whole img
            gt_bx_h = (box_ops.box_cxcywh_to_xyxy(targets['boxes_h']) * scale_fct[i][None,:]).half()
            gt_bx_o = (box_ops.box_cxcywh_to_xyxy(targets['boxes_o']) * scale_fct[i][None,:]).half()
            verbs = targets['labels']
            hois = targets['hoi'] ## todo-hoi
            filename = targets['filename']
            objects_label = targets['object']
            lt = torch.min(gt_bx_h[..., :2], gt_bx_o[..., :2]) # left point
            rb = torch.max(gt_bx_h[..., 2:], gt_bx_o[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1).half()
            
            if len(targets['boxes_h']) == 0:
                print(targets['filename'])
                pdb.set_trace()
            lens = local_features.shape[0]
            huamn_features = local_features[:lens//3,:]
            object_features = local_features[lens//3:lens//3*2,:]
            union_features = local_features[lens//3*2:,:]
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # pdb.set_trace()
            size = targets['orig_size']
            self.dicts[filename] = dict(boxes_h=gt_bx_h.cpu().numpy(),
            boxes_o=gt_bx_o.cpu().numpy(), verbs=verbs.cpu().numpy(), hois=hois.cpu().numpy(), ## todo-hoi
            union_boxes=union_boxes.cpu().numpy(), union_features=union_features.cpu().numpy(),
            huamn_features=huamn_features.cpu().numpy(), object_features=object_features.cpu().numpy(),
            objects=objects_label.cpu().numpy(), global_feature = global_feature.cpu().numpy(),
            size_orig=size.cpu().numpy()
            )
            
        return 

    def compute_roi_embeddings_targets_crop_invalid(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict], targets: List[dict]):
        device = features.device
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        for i, props in enumerate(region_props):
            local_features = features[i]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            filename = targets[i]['filename']
            verbs = targets[i]['labels']
            hois = targets[i]['hoi']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1: continue

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
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            

            # compute predict bbox with the gts
            
            gt_bx_h = self.recover_boxes(targets[i]['boxes_h'], targets[i]['size'])
            gt_bx_o = self.recover_boxes(targets[i]['boxes_o'], targets[i]['size'])
            # pdb.set_trace()
            x, y = torch.nonzero(torch.min(
                box_iou(sub_boxes, gt_bx_h),
                box_iou(obj_boxes, gt_bx_o)
            ) >= self.fg_iou_thresh).unbind(1)
            all_pairs = torch.arange(len(sub_boxes))
            invalid_indexes = torch.as_tensor(list(set(all_pairs.numpy()) - set(x.cpu().numpy())))
            if len(invalid_indexes) ==0 : continue


            # verbs = verbs[y]
            # hois = hois[y]
            union_boxes = torch.cat([lt,rb],dim=-1)[invalid_indexes]
            sub_boxes = sub_boxes[invalid_indexes]
            obj_boxes = obj_boxes[invalid_indexes]
            lens = local_features.shape[0]
            huamn_features = local_features[:lens//3,:][invalid_indexes]
            object_features = local_features[lens//3:lens//3*2,:][invalid_indexes]
            union_features = local_features[lens//3*2:,:][invalid_indexes]
            # pdb.set_trace()
            assert union_features.shape == huamn_features.shape == object_features.shape
            self.dicts[filename] = dict(boxes_h=sub_boxes.cpu().numpy(),
            boxes_o=obj_boxes.cpu().numpy(),
            union_boxes=union_boxes.cpu().numpy(), union_features=union_features.cpu().numpy(),
            huamn_features=huamn_features.cpu().numpy(), object_features=object_features.cpu().numpy(),
            objects=labels[y_keep][invalid_indexes].cpu().numpy()
            )
        return 

    def compute_roi_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        raise AssertionError
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []
        text_embeddings_bs = self.text_embedding
        text_embeddings_bs = text_embeddings_bs / text_embeddings_bs.norm(dim=-1, keepdim=True)
        # text_embeddings = self.beta * self.adapter_t(text_embeddings) + (1-self.beta) * text_embeddings
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            # local_features = features[:,b_idx,:]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']
            # add mask
            masks = props['mask']
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
            
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1).half()
            
            # union_mask = masks[x_keep] | masks[y_keep]
            
            # pdb.set_trace()
            ### DEFR
            # union_mask = torch.ones((union_boxes.shape[0],7,7),dtype=torch.bool,device=union_boxes.device)
            # # pdb.set_trace()
            # for i in range(len(union_boxes)):
            #     union_box = (union_boxes[i]//32).int()
            #     union_mask[i,union_box[1]:union_box[3],union_box[0]:union_box[2]]= False
            # union_mask = torch.cat([torch.ones(len(union_mask),dtype=torch.bool).unsqueeze(1).to(device),union_mask.flatten(1)],dim=-1)
            # x = local_features.unsqueeze(1).repeat(1,len(union_mask),1)
            # x = self.clip_model.visual.transformer.resblocks[-1]((x,union_mask))[0]
            # x = x.permute(1, 0, 2)  # LND -> NLD
            # x = self.clip_model.visual.ln_post(x[:, 0, :])
            # union_features = x @ self.clip_model.visual.proj

            ### weighted union mask 
            # union_mask = masks[x_keep] | masks[y_keep]
            # union_boxes_mask = torch.cat([torch.arange(len(x_keep)).unsqueeze(1).to(device),union_mask],dim=-1)
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            # union_mask_weights = torchvision.ops.roi_align(union_mask.unsqueeze(1),union_boxes_mask,output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            # union_mask_weights = union_mask_weights.flatten(2).softmax(-1).view(union_mask.shape[0],1,7,7)
            # union_features = (union_features * union_mask_weights).flatten(2).sum(-1)
            
            ### rectangle kernel
            sub_box_cxcy = box_xyxy_to_cxcywh(sub_boxes)
            obj_box_cxcy = box_xyxy_to_cxcywh(obj_boxes)
            dist = ((sub_box_cxcy[:,0] - obj_box_cxcy[:,0]).pow(2) + (sub_box_cxcy[:,1] - obj_box_cxcy[:,1]).pow(2)).sqrt()/2
            mask = torch.zeros((len(union_boxes), 224, 224), dtype=torch.bool)
            verb_cor = torch.cat(((sub_box_cxcy[:,0] + obj_box_cxcy[:,0]).unsqueeze(1),(sub_box_cxcy[:,1] + obj_box_cxcy[:,1]).unsqueeze(1)),dim=-1)/2
            verb_cors = box_cxcywh_to_xyxy(torch.cat((verb_cor,dist.unsqueeze(1),dist.unsqueeze(1)),dim=-1))

            # pdb.set_trace()
            union_mask = torch.zeros((union_boxes.shape[0],224,224),dtype=torch.bool,device=union_boxes.device)
            for i in range(len(union_boxes)):
                sub_box = sub_boxes[i].int()
                obj_box = obj_boxes[i].int()
                # union_box = union_boxes[i].int()
                verb_cor = verb_cors[i].int()
                union_mask[i,verb_cor[1]:verb_cor[3],verb_cor[0]:verb_cor[2]]= True
                union_mask[i,sub_box[1]:sub_box[3],sub_box[0]:sub_box[2]]= True
                union_mask[i,obj_box[1]:obj_box[3],obj_box[0]:obj_box[2]]= True
            union_mask = F.interpolate(union_mask[None].float(), size=(7,7)).to(torch.bool)[0]
            union_boxes_mask = torch.cat([torch.arange(len(union_mask)).unsqueeze(1).to(device),union_boxes],dim=-1)
            
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            union_mask_weights = torchvision.ops.roi_align(union_mask.half().unsqueeze(1),union_boxes_mask,output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True)
            union_features = (union_features * union_mask_weights).flatten(2).sum(-1)

            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            
            logit_scale = self.logit_scale.exp()
            union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            # pdb.set_trace()
            # text_embeddings_bs = text_embeddings[b_idx] / text_embeddings[b_idx].norm(dim=-1, keepdim=True)
            # logits = logit_scale * self.visual_projection(union_features)
            # logits = logit_scale * union_features @ text_embeddings_bs.t()
            logits =  union_features @ text_embeddings_bs.t()
            all_logits.append(logits.float())
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
        # pdb.set_trace()
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        
        labels[x, targets['labels'][y]] = 1
        # labels[x, targets['hoi'][y]] = 1
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
        # try:
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
            
        # except:
        #     pdb.set_trace()
        # if loss.isnan():
        #     pdb.set_trace()
        # loss = binary_focal_loss_with_logits(
        #     logits, labels, reduction='sum',
        #     alpha=self.alpha, gamma=self.gamma
        # )
        # print(loss)
        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states):
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)

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

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
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
            # scores = lg[x, y]
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections
    
    def get_region_proposals(self, results):
        region_props = []
        for res in results:
            # pdb.set_trace()
            bx = res['ex_bbox']
            sc = res['ex_scores']
            lb = res['ex_labels']
            hs = res['ex_hidden_states']
            ms = res['ex_mask']
            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human.sum(); n_object = len(lb) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                # keep_h = keep[keep_h]
                keep_h = hum

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                # keep_o = keep[keep_o]
                keep_o = obj

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep],
                mask = ms[keep]
            ))

        return region_props
        
    def get_targets_pairs(self, targets):
        region_targets = {}
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        for tar in targets:
            # pdb.set_trace()
            gt_bx_h = self.recover_boxes(tar['boxes_h'], tar['size'])
            gt_bx_o = self.recover_boxes(tar['boxes_o'], tar['size'])
            verbs = tar['labels']
            filename = tar['filename']
            region_targets['filename'] = dict(
                boxes_h=gt_bx_h,
                boxes_o=gt_bx_o,
                verbs=verbs,
            )

        return region_targets
    
    def get_prior(self, region_props,image_size):
        
        max_feat = self.visual_output_dim + 5
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_w, img_h = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                # pdb.set_trace()
                print(n_h,n)
                # sys.exit()
            # pdb.set_trace()
            object_embs = self.object_embedding[labels]
            priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            priors[b_idx,:n,5:] = object_embs
            mask[b_idx,:n] = False
        # pdb.set_trace()
        priors = self.priors(priors)
        return (priors, mask)

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

        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)
        # region_props = self.get_region_proposals(targets) 
        # feat_local = self.get_clip_feature(images_clip.decompose()[0])
        # feat_local_old = self.clip_model.encode_image(images[0])
        feat_local_old = forward_chunks(self.clip_model.encode_image, images[0])
        # feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 14, 14)
        global_feat = feat_local_old[:,0,:]
        # self.compute_roi_embeddings_targets_crop_invalid(global_feat.unsqueeze(0), image_sizes, region_props, targets)
        self.compute_roi_embeddings_targets_crop(global_feat.unsqueeze(0), image_sizes, targets)
        return None

        boxes = [r['boxes'] for r in region_props]
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

def build_detector(args, class_corr):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    interaction_head = InteractionHead(
        predictor, args.hidden_dim, args.repr_dim,
        detr.backbone[0].num_channels,
        args.num_classes, args.human_idx, class_corr
    )
    
    if args.visual_mode == 'res':
        clip_head = CLIP_ResNet(embed_dim=args.clip_visual_output_dim,
                                image_resolution=args.clip_visual_input_resolution,
                                vision_layers=args.clip_visual_layers,
                                vision_width=args.clip_visual_width,
                                vision_patch_size=args.clip_visual_patch_size,
                                context_length=args.clip_text_context_length,
                                transformer_width=args.clip_text_transformer_width,
                                transformer_heads=args.clip_text_transformer_heads,
                                transformer_layers=args.clip_text_transformer_layers)
    elif args.visual_mode == 'vit':
        clip_head = CLIP_ResNet(embed_dim=args.clip_visual_output_dim_vit,
                                image_resolution=args.clip_visual_input_resolution_vit,
                                vision_layers=args.clip_visual_layers_vit,
                                vision_width=args.clip_visual_width_vit,
                                vision_patch_size=args.clip_visual_patch_size_vit,
                                context_length=args.clip_text_context_length_vit,
                                transformer_width=args.clip_text_transformer_width_vit,
                                transformer_heads=args.clip_text_transformer_heads_vit,
                                transformer_layers=args.clip_text_transformer_layers_vit)
    detector = UPT(args,
        detr, postprocessors['bbox'], clip_head, args.clip_dir_vit,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
    )
    return detector
