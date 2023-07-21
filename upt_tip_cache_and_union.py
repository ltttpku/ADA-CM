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
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from hico_text_label import hico_text_label, hico_obj_text_label
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

import pdb
# from CLIP_models import CLIP_ResNet, tokenize
from CLIP_models_adapter_prior import CLIP_ResNet, tokenize
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle
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

        # self.clip_head.init_weights(clip_pretrained)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        # self.clip_model, _ = clip.load("RN50", device=device)
        # pdb.set_trace()
        self.class_nums = 600
        use_templates = False
        if self.class_nums==117 :text_inputs = torch.cat([clip.tokenize(verb) for verb in hico_verbs]) # hico_verbs 'action is ' +
        elif self.class_nums==600 and use_templates==False:            
            text_inputs = torch.cat([clip.tokenize(' '.join(hico_text_label[id].split(' ')[3:])) for id in hico_text_label.keys()])
        elif self.class_nums==600 and use_templates==True:
            text_inputs = self.get_multi_prompts(hico_text_label)
            bs_t, nums, c = text_inputs.shape
            text_inputs = text_inputs.view(-1, c)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs.to(device))
        if use_templates:
            text_embedding = text_embedding.view(bs_t, nums, -1).mean(0)

        hico_triplet_labels = list(hico_text_label.keys())
        hoi_obj_list = []
        for hoi_pair in hico_triplet_labels:
            hoi_obj_list.append(hoi_pair[1])
        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in hico_obj_text_label])

        with torch.no_grad():
            obj_text_embedding = self.clip_model.encode_text(obj_text_inputs.to(device))[hoi_obj_list,:]

        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        obj_text_embedding = obj_text_embedding / obj_text_embedding.norm(dim=-1, keepdim=True)
        self.obj_hoi_embedding = torch.cat([obj_text_embedding[0].unsqueeze(0).repeat(600,1), obj_text_embedding,text_embedding],dim=-1)

        self.logit_scale = self.clip_model.logit_scale
        self.text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
       
        
        self.dicts = {}
        self.feature = 'union' # union, obj_uni, hum_obj_uni
        
        # num_shot = 32
        # save_file2 = 'save_sample_indexes_{}_{}_clipcrop.p'.format(self.class_nums,num_shot)
        
        # self.hois_cooc = torch.load('one_hots.pt')

        # self.cache_models, self.one_hots, self.sample_lens = self.load_cache_model(file1='union_embeddings_cachemodel_clipcrops.p',file2=save_file2, feature=self.feature,class_nums=self.class_nums, num_shot=num_shot)
        
        # self.cache_models = (self.cache_models / self.cache_models.norm(dim=-1, keepdim=True)).cuda().float()
        # self.one_hots = self.one_hots.cuda().float()
        # self.beta_cache = torch.tensor(10) 
        # self.alpha_cache = torch.tensor(1.0)
        # self.sample_lens = torch.as_tensor(self.sample_lens).cuda()
    
    def get_multi_prompts(self, hico_labels):
        templates = ['itap of {}', '“a bad photo of {}', 'a photo of {}', 'there is {} in the video game', 'art of {}', 'the picture describes {}']
        hico_texts = [hico_text_label[id].split(' ')[3:] for id in hico_text_label.keys()]
        all_texts_input = []
        for temp in templates:
            texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
            all_texts_input.append(texts_input)
        all_texts_input = torch.stack(all_texts_input,dim=0)
        return all_texts_input
    def load_cache_model(self,file1, file2=None, category='verb', feature='union',class_nums=117, num_shot=10):
        
        annotation = pickle.load(open(file1,'rb'))
        # if category == 'verb':
        categories = class_nums
        union_embeddings = [[] for i in range(categories)]
        obj_embeddings = [[] for i in range(categories)]
        hum_embeddings = [[] for i in range(categories)]
        filenames = list(annotation.keys())
        
        for file_n in filenames:
            anno = annotation[file_n]
            if categories == 117: verbs = anno['verbs']
            else: verbs = anno['hois']
            
            union_features = anno['union_features']
            object_features = anno['object_features']
            # pdb.set_trace()
            huamn_features = anno['huamn_features']
            if len(verbs) == 0:
                print(file_n)
            for i, v in enumerate(verbs):
                union_embeddings[v].append(union_features[i])
                obj_embeddings[v].append(object_features[i])
                hum_embeddings[v].append(huamn_features[i])
                
        all_lens = torch.as_tensor([len(u) for u in union_embeddings])
        K_shot = num_shot
        K_sample_lens = all_lens>K_shot
        all_lens_sample = torch.ones(categories) * K_shot
        index_lessK = torch.where(K_sample_lens==False)[0]
        for index in index_lessK:
            all_lens_sample[index] = len(union_embeddings[index])
        all_lens_sample = torch.cumsum(all_lens_sample,dim=-1).long()
        if file2 is not None:
            save_sample_index = pickle.load(open(file2, 'rb'))
        else:
            save_sample_index = []
        
        if feature == 'union' :
          
            cache_models = torch.zeros((all_lens_sample[-1], 512),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            for i, embeddings in enumerate(union_embeddings):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) >= K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot)
                    sample_embeddings = np.array(embeddings)[sample_index]
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = np.array(embeddings)
                if i==0:
                    cache_models[:all_lens_sample[i],:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot
                    # one_hot = self.hois_cooc[i][None,:].repeat(lens,1)
                    # one_hots[:all_lens_sample[i],:] = one_hot

                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot
                    # one_hot = self.hois_cooc[i][None,:].repeat(lens,1)
                    # one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot
                    
                each_lens.append(lens)
                save_sample_index.append(sample_index)
        elif feature == 'obj_uni':
            cache_models = torch.zeros((all_lens_sample[-1], 512*2),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            indexes = np.arange(len(union_embeddings))
            for i, obj_emb, embeddings in zip(indexes, obj_embeddings, union_embeddings):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) >= K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot)
                    sample_embeddings = np.array(embeddings)[sample_index]
                    sample_obj_embeddings =  np.array(obj_emb)[sample_index]
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = np.array(embeddings)
                    sample_obj_embeddings =  np.array(obj_emb)
                if i==0:
                    cache_models[:all_lens_sample[i],:512] = torch.as_tensor(sample_obj_embeddings)
                    cache_models[:all_lens_sample[i],512:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot
                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:512] = torch.as_tensor(sample_obj_embeddings)
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],512:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot
                each_lens.append(lens)
                save_sample_index.append(sample_index)
        elif feature == 'hum_obj_uni':
            cache_models = torch.zeros((all_lens_sample[-1], 512*3),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings in zip(indexes, hum_embeddings, obj_embeddings, union_embeddings):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) >= K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot)
                    sample_embeddings = np.array(embeddings)[sample_index]
                    sample_obj_embeddings =  np.array(obj_emb)[sample_index]
                    sample_hum_embeddings =  np.array(hum_emb)[sample_index]
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = np.array(embeddings)
                    sample_obj_embeddings = np.array(obj_emb)
                    sample_hum_embeddings = np.array(hum_emb)
                if i==0:
                    cache_models[:all_lens_sample[i],:512] = torch.as_tensor(sample_hum_embeddings)
                    cache_models[:all_lens_sample[i],512:1024] = torch.as_tensor(sample_obj_embeddings)
                    cache_models[:all_lens_sample[i],1024:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot
                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:512] = torch.as_tensor(sample_hum_embeddings)
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],512:1024] = torch.as_tensor(sample_obj_embeddings)
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],1024:] = torch.as_tensor(sample_embeddings)
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot
                each_lens.append(lens)
                save_sample_index.append(sample_index)
                
        return cache_models,one_hots, each_lens
        
        # elif category == 'hoi':

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
        # p = 1.0
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
    def compute_roi_embeddings_targets(self, features: OrderedDict, image_shapes: Tensor, targets_region_props: List[dict]):

        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        all_logits = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_boxes = []
        for i, targets in enumerate(targets_region_props):
            # pdb.set_trace()
            local_features = features[i]
            gt_bx_h = (box_ops.box_cxcywh_to_xyxy(targets['boxes_h']) * scale_fct[i][None,:])
            gt_bx_o = (box_ops.box_cxcywh_to_xyxy(targets['boxes_o']) * scale_fct[i][None,:])
            verbs = targets['labels']
            hois = targets['hoi']
            filename = targets['filename']
            objects_label = targets['object']
            lt = torch.min(gt_bx_h[..., :2], gt_bx_o[..., :2]) # left point
            rb = torch.max(gt_bx_h[..., 2:], gt_bx_o[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            
            # pdb.set_trace()
            # concat_feat = torch.cat([huamn_features,object_features, union_features],dim=-1)
            # concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
            # logits = concat_feat @ self.obj_hoi_embedding.t()

            # logits = logits.softmax(-1)
            union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            logits = union_features @ self.text_embedding.t()

            
            logits_cache = ((union_features @ self.cache_models.t()) @ self.one_hots) / self.sample_lens
            logits = logits + logits_cache
            # pdb.set_trace()
            all_boxes.append(torch.cat([gt_bx_h,gt_bx_o],dim=0))
            keep = torch.arange(len(gt_bx_h)*2).to(union_features.device)
            boxes_h_collated.append(keep[:len(gt_bx_h)])
            boxes_o_collated.append(keep[len(gt_bx_h):])
            object_class_collated.append(objects_label)
            scores = torch.ones(len(keep)).to(union_features.device)
            
            prior_collated.append(self.compute_prior_scores(
                keep[:len(gt_bx_h)], keep[:len(gt_bx_o)], scores, objects_label)
            )
            
            # self.dicts[filename] = dict(boxes_h=gt_bx_h.cpu().numpy(),
            # boxes_o=gt_bx_o.cpu().numpy(), verbs=verbs.cpu().numpy(), hois=hois.cpu().numpy(),
            # union_boxes=union_boxes.cpu().numpy(), union_features=union_features.cpu().numpy(),
            # huamn_features=huamn_features.cpu().numpy(), object_features=object_features.cpu().numpy(),
            # objects=objects_label
            # )
            all_logits.append(logits.float())
        return all_logits,prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated ,all_boxes

    def compute_roi_embeddings_returnbx(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []
        # text_embeddings_bs = self.text_embedding
        # text_embeddings_bs = text_embeddings_bs / text_embeddings_bs.norm(dim=-1, keepdim=True)
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
            # union_boxes = torch.cat([lt,rb],dim=-1).half()
            # lens = local_features.shape[0]
            # human_features = local_features[:lens//3,:]
            # object_features = local_features[lens//3:lens//3*2,:]
            # union_features = local_features[lens//3*2:,:]
            # pdb.set_trace()
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes.half()],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).mean(3).mean(2)
            # human_features = single_features[x_keep]
            # object_features = single_features[y_keep]

            # union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            # if self.feature == 'union' :
            #     # pdb.set_trace()
            #     union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            #     phi_union = torch.exp(-self.beta_cache*(middle_point-(union_features@self.cache_models.t())))
            #     # phi_union = -self.beta_cache*(middle_point-(union_features@self.cache_models.t()))
            #     # phi_union = -self.beta_cache*(middle_point-(union_features@self.cache_models.t()))
            # elif self.feature == 'obj_uni' :
            #     concat_feat = torch.cat([object_features, union_features],dim=-1)
            #     concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
            #     phi_union = torch.exp(-self.beta_cache*(middle_point-(concat_feat@self.cache_models.t())))
            # elif self.feature == 'hum_obj_uni' :
            #     # pdb.set_trace()
            #     concat_feat = torch.cat([human_features,object_features, union_features],dim=-1)
            #     concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
            #     phi_union = torch.exp(-self.beta_cache*(middle_point-(concat_feat@self.cache_models.t())))

            # else :
            #     raise ValueError(f'unknown {self.feature}')
            
            # logits_cache = self.alpha_cache * (phi_union @ self.one_hots) /self.sample_lens  ###   self.sample_lens
            # logits = logits_cache
            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            
        return prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated


    def compute_roi_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []
        # text_embeddings_bs = self.text_embedding
        # text_embeddings_bs = text_embeddings_bs / text_embeddings_bs.norm(dim=-1, keepdim=True)
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
            # union_boxes = torch.cat([lt,rb],dim=-1).half()
            lens = local_features.shape[0]
            human_features = local_features[:lens//3,:]
            object_features = local_features[lens//3:lens//3*2,:]
            union_features = local_features[lens//3*2:,:]
            # pdb.set_trace()
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes.half()],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).mean(3).mean(2)
            # human_features = single_features[x_keep]
            # object_features = single_features[y_keep]

            # union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            middle_point = 1
            if self.feature == 'union' :
                # pdb.set_trace()
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                phi_union = torch.exp(-self.beta_cache*(middle_point-(union_features@self.cache_models.t())))
                # phi_union = -self.beta_cache*(middle_point-(union_features@self.cache_models.t()))
                # phi_union = -self.beta_cache*(middle_point-(union_features@self.cache_models.t()))
            elif self.feature == 'obj_uni' :
                concat_feat = torch.cat([object_features, union_features],dim=-1)
                concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
                phi_union = torch.exp(-self.beta_cache*(middle_point-(concat_feat@self.cache_models.t())))
            elif self.feature == 'hum_obj_uni' :
                # pdb.set_trace()
                concat_feat = torch.cat([human_features,object_features, union_features],dim=-1)
                concat_feat = concat_feat / concat_feat.norm(dim=-1, keepdim=True)
                phi_union = torch.exp(-self.beta_cache*(middle_point-(concat_feat@self.cache_models.t())))

            else :
                raise ValueError(f'unknown {self.feature}')
            
            # union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            logits_cache = self.alpha_cache * (phi_union @ self.one_hots) /self.sample_lens  ###   self.sample_lens
            # pdb.set_trace()
            # logits_test = union_features @ self.text_embedding.t()
            # # # logits = logits_cache
            # logits = logits_cache + logits_test
            # logits = logits
            logits = logits_cache


            # logits = self.visual_classify(concat_feat)
            # pdb.set_trace()

            # logits = logits_test.sigmoid()
            
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            
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
            # scores = torch.sigmoid(lg[x, y])
            scores = lg[x, y]
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
        
        max_feat = 512 + 5
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

        ### roi + gts
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        ######   
        pdb.set_trace()
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)
        
        # feat_local_old = self.clip_model.encode_image(images[0]).float()
        # cls_feature = self.clip_model.encode_image(images[0]).float()
        # logits, prior, bh, bo, objects = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
        feat_local_old = self.clip_model.encode_image(images[0].unsqueeze(0)).float()
        feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 7, 7)
        logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(feat_local, image_sizes, targets)

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

        # global_feat = feat_local_old[:,0,:]
        # # global_feat = global_feat/global_feat.norm(dim=-1, keepdim=True)
        # # logits = global_feat @ self.text_embedding.t()
        # lens = global_feat.shape[0]
        # # pdb.set_trace()
        # feat1 = global_feat[:lens//3,:]
        # feat2 = global_feat[lens//3:lens//3*2,:]
        # feat3 = global_feat[lens//3*2:,:]
        # feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        # feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        # feat3 = feat3 / feat3.norm(dim=-1, keepdim=True)
        # concat_feat = torch.cat([feat1,feat2,feat3],dim=-1)
        # concat_feat = torch.cat([global_feat[:lens//2,:],global_feat[:lens//2,:]],dim=-1)
        # concat_feat = torch.cat([global_feat[:lens//3,:],global_feat[lens//3:lens//3*2,:],global_feat[lens//3*2:,:]],dim=-1)
        # concat_feat = concat_feat/concat_feat.norm(dim=-1, keepdim=True)


        # # logits = concat_feat @ self.obj_hoi_embedding.t()
        # # global_feat = global_feat/global_feat.norm(dim=-1, keepdim=True)
        # # pdb.set_trace()
        # logits = feat3 @ self.text_embedding.t().float()
        # logits = logits
        # # logits_cache = ((feat3 @ self.cache_models.t()) @ self.one_hots) /self.sample_lens

        # # logits = logits + 0.5 * logits_cache
        # detections = self.postprocessing(boxes, bh, bo, [logits], prior, objects, image_sizes)
        # return detections

        #@@@@ extract box 
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        ######
        # pdb.set_trace()
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)
        
        
        feat_local_old = self.clip_model.encode_image(images[0])
        
        # feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 7, 7)
        feat_local = feat_local_old
        # feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 14, 14)
        region_props = self.get_region_proposals(targets)
        prior, bh, bo, objects = self.compute_roi_embeddings_returnbx(feat_local, image_sizes, region_props)
        # logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(feat_local, image_sizes, targets)
        # global_feat = feat_local_old[:,0,:]
        # global_feat = feat_local_old
        global_feat = feat_local_old[:,0,:]
        global_feat = global_feat/global_feat.norm(dim=-1, keepdim=True)
        lens = global_feat.shape[0]
        # # concat_feat = torch.cat([global_feat[:lens//2,:],global_feat[:lens//2,:]],dim=-1)
        # # concat_feat = concat_feat/concat_feat.norm(dim=-1, keepdim=True)
        # # feat1 = global_feat[:lens//2,:]
        # # feat2 = global_feat[lens//2:,:]
        # # feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        # # feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        # # concat_feat = torch.cat([feat1,feat2],dim=-1)
        # # concat_feat = concat_feat/concat_feat.norm(dim=-1, keepdim=True)
        # # pdb.set_trace()q
        # # concat_feat = torch.cat([global_feat[:lens//3,:],global_feat[lens//3:lens//3*2,:],global_feat[lens//3*2:,:]],dim=-1)
        # # concat_feat = concat_feat/concat_feat.norm(dim=-1, keepdim=True)
        # # pdb.set_trace()
        # # logits = concat_feat @ self.obj_hoi_embedding.t()
        # # logits = global_feat @ self.text_embedding.t()
        feat1 = global_feat[:lens//3,:]
        feat2 = global_feat[lens//3:lens//3*2,:]
        feat3 = global_feat[lens//3*2:,:]
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        feat3 = feat3 / feat3.norm(dim=-1, keepdim=True)
        # concat_feat = torch.cat([feat1,feat2,feat3],dim=-1)
        # concat_feat = torch.cat([global_feat[:lens//2,:],global_feat[:lens//2,:]],dim=-1)
        # concat_feat = torch.cat([global_feat[:lens//3,:],global_feat[lens//3:lens//3*2,:],global_feat[lens//3*2:,:]],dim=-1)
        # concat_feat = concat_feat/concat_feat.norm(dim=-1, keepdim=True)
        # pdb.set_trace()
        
        # logits = global_feat @ self.text_embedding.t()
        # pdb.set_trace()
        logits = feat3 @ self.text_embedding.t().float()
        logits = logits
        # logits_cache = ((feat3 @ self.cache_models.t()) @ self.one_hots) /self.sample_lens

        # logits = logits + 0.5 * logits_cache

        # pdb.set_trace()
        # logits = logits.sigmoid()
        boxes = [r['boxes'] for r in region_props]
        detections = self.postprocessing(boxes, bh, bo, [logits], prior, objects, image_sizes)
        return detections



        #######
        
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images
        ], device=images[0].device)
        
        # if isinstance(images, (list, torch.Tensor)):
        #     images_clip = nested_tensor_from_tensor_list(images)
        region_props = self.get_region_proposals(targets)

        # feat_local_old = self.clip_model.encode_image(images_clip.decompose()[0])
        feat_local_old = self.clip_model.encode_image(images[0])
        feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 7, 7)
        global_feat = feat_local_old[:,0,:]
        
        # global_feat = global_feat/global_feat.norm(dim=-1, keepdim=True)
        # logits_g = global_feat @ self.text_embedding.t()
        # pdb.set_trace()
        logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(feat_local, image_sizes, targets)
        # pdb.set_trace()
        # logits, prior, bh, bo, objects = self.compute_roi_embeddings(global_feat.unsqueeze(0), image_sizes, region_props)
        
        

        boxes = [r['boxes'] for r in region_props]
        # pdb.set_trace()
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
    detector = UPT(
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
