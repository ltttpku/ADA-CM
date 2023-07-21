"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from __future__ import annotations
from difflib import unified_diff
import os
import random
from turtle import position
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits
from interaction_head import InteractionHead

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
# from hico_text_label import hico_text_label, hico_obj_text_label
from vcoco_list import vcoco_verbs_sentence

sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from d_detr.models import build_model as build_model_d_detr
import PIL
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
from tqdm import tqdm
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from tools import forward_chunks
from vcoco_text_label import vcoco_hoi_text_label
import hico_text_label

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
        object_class_to_target_class: List[list] = None, 
        topk: int = 250,
        **kwargs,
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
        self.num_anno = kwargs["num_anno"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_process = clip.load(args.clip_model_name, device=device)

        if self.num_classes == 117:
            verb_lst = hico_verbs_sentence
        elif self.num_classes == 24:
            verb_lst = vcoco_verbs_sentence
        self.class_nums = num_classes
        use_templates = False
        if self.class_nums==117 or self.class_nums==24:
            text_inputs = torch.cat([clip.tokenize(verb) for verb in verb_lst])
        elif self.class_nums==600 and use_templates==False:
            text_inputs = torch.cat([clip.tokenize(hico_text_label.hico_text_label[id]) for id in hico_text_label.hico_text_label.keys()])
        elif self.class_nums==236 and use_templates==False:
            text_inputs = torch.cat([clip.tokenize(vcoco_hoi_text_label[id]) for id in vcoco_hoi_text_label.keys()])
        elif self.class_nums==600 and use_templates==True:
            text_inputs = self.get_multi_prompts(hico_text_label)
            bs_t, nums, c = text_inputs.shape
            text_inputs = text_inputs.view(-1, c)

        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs.to(device))
        if use_templates:
            text_embedding = text_embedding.view(bs_t, nums, -1).mean(0)

        # use object embedding
        self.dataset = args.dataset
        if args.dataset == 'hico' or args.dataset == 'vcoco':
            hico_triplet_labels = list(hico_text_label.hico_text_label.keys())
            hoi_obj_list = []
            for hoi_pair in hico_triplet_labels:
                hoi_obj_list.append(hoi_pair[1])

        self.text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 600*512
        self.text_embedding = self.text_embedding.float()

        if 'HO+U' in args.logits_type or 'HO+T' in args.logits_type:
            self.feature = 'hum_obj_uni'
        elif 'HO' in args.logits_type:
            self.feature = 'hum_obj'
        else:
            self.feature = 'uni'
        self.logits_type = args.logits_type

        self.HOI_IDX_TO_ACT_IDX = [
            4, 17, 25, 30, 41, 52, 76, 87, 111, 57, 8, 36, 41, 43, 37, 62, 71, 75, 76,
            87, 98, 110, 111, 57, 10, 26, 36, 65, 74, 112, 57, 4, 21, 25, 41, 43, 47,
            75, 76, 77, 79, 87, 93, 105, 111, 57, 8, 20, 36, 41, 48, 58, 69, 57, 4, 17,
            21, 25, 41, 52, 76, 87, 111, 113, 57, 4, 17, 21, 38, 41, 43, 52, 62, 76,
            111, 57, 22, 26, 36, 39, 45, 65, 80, 111, 10, 57, 8, 36, 49, 87, 93, 57, 8,
            49, 87, 57, 26, 34, 36, 39, 45, 46, 55, 65, 76, 110, 57, 12, 24, 86, 57, 8,
            22, 26, 33, 36, 38, 39, 41, 45, 65, 78, 80, 98, 107, 110, 111, 10, 57, 26,
            33, 36, 39, 43, 45, 52, 37, 65, 72, 76, 78, 98, 107, 110, 111, 57, 36, 41,
            43, 37, 62, 71, 72, 76, 87, 98, 108, 110, 111, 57, 8, 31, 36, 39, 45, 92,
            100, 102, 48, 57, 8, 36, 38, 57, 8, 26, 34, 36, 39, 45, 65, 76, 83, 110,
            111, 57, 4, 21, 25, 52, 76, 87, 111, 57, 13, 75, 112, 57, 7, 15, 23, 36,
            41, 64, 66, 89, 111, 57, 8, 36, 41, 58, 114, 57, 7, 8, 15, 23, 36, 41, 64,
            66, 89, 57, 5, 8, 36, 84, 99, 104, 115, 57, 36, 114, 57, 26, 40, 112, 57,
            12, 49, 87, 57, 41, 49, 87, 57, 8, 36, 58, 73, 57, 36, 96, 111, 48, 57, 15,
            23, 36, 89, 96, 111, 57, 3, 8, 15, 23, 36, 51, 54, 67, 57, 8, 14, 15, 23,
            36, 64, 89, 96, 111, 57, 8, 36, 73, 75, 101, 103, 57, 11, 36, 75, 82, 57,
            8, 20, 36, 41, 69, 85, 89, 27, 111, 57, 7, 8, 23, 36, 54, 67, 89, 57, 26,
            36, 38, 39, 45, 37, 65, 76, 110, 111, 112, 57, 39, 41, 58, 61, 57, 36, 50,
            95, 48, 111, 57, 2, 9, 36, 90, 104, 57, 26, 45, 65, 76, 112, 57, 36, 59,
            75, 57, 8, 36, 41, 57, 8, 14, 15, 23, 36, 54, 57, 8, 12, 36, 109, 57, 1, 8,
            30, 36, 41, 47, 70, 57, 16, 36, 95, 111, 115, 48, 57, 36, 58, 73, 75, 109,
            57, 12, 58, 59, 57, 13, 36, 75, 57, 7, 15, 23, 36, 41, 64, 66, 91, 111, 57,
            12, 36, 41, 58, 75, 59, 57, 11, 63, 75, 57, 7, 8, 14, 15, 23, 36, 54, 67,
            88, 89, 57, 12, 36, 56, 58, 57, 36, 68, 99, 57, 8, 14, 15, 23, 36, 54, 57,
            16, 36, 58, 57, 12, 75, 111, 57, 8, 28, 32, 36, 43, 67, 76, 87, 93, 57, 0,
            8, 36, 41, 43, 67, 75, 76, 93, 114, 57, 0, 8, 32, 36, 43, 76, 93, 114, 57,
            36, 48, 111, 85, 57, 2, 8, 9, 19, 35, 36, 41, 44, 67, 81, 84, 90, 104, 57,
            36, 94, 97, 57, 8, 18, 36, 39, 52, 58, 60, 67, 116, 57, 8, 18, 36, 41, 43,
            49, 52, 76, 93, 87, 111, 57, 8, 36, 39, 45, 57, 8, 36, 41, 99, 57, 0, 15,
            36, 41, 70, 105, 114, 57, 36, 59, 75, 57, 12, 29, 58, 75, 87, 93, 111, 57,
            6, 36, 111, 57, 42, 75, 94, 97, 57, 17, 21, 41, 52, 75, 76, 87, 111, 57, 8,
            36, 53, 58, 75, 82, 94, 57, 36, 54, 61, 57, 27, 36, 85, 106, 48, 111, 57,
            26, 36, 65, 112, 57
        ]

        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]
        
        self.filtered_idx = hico_text_label.hico_unseen_index['non_rare_first']
        self._load_features = False

        self.temperature = args.temperature
        num_shot = args.num_shot
        self.gamma_HO = args.gamma_HO
        self.gamma_U = args.gamma_U

        file1 = args.file1
        print('[INFO]: Traning-Free mode')

        self.use_multi_hot = args.use_multi_hot
        self.label_choice = args.label_choice
        self.rm_duplicate_feat = args.rm_duplicate_feat
        self.sample_choice = args.sample_choice
        self.dic_key = args.dic_key
        print('>>>>> ------------------ origin load cache model! ------------------ <<<<<<<<<<<<<<<<<<<<<')
        if 'HO' in self.logits_type:
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.load_cache_model(file1=file1, feature='hum_obj',class_nums=self.class_nums, num_shot=num_shot, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO  = self.cache_model_HO.cuda().float(), self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()
        if 'U' in self.logits_type:
            self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.load_cache_model(file1=file1, feature='uni',class_nums=self.class_nums, num_shot=num_shot, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.cache_model_U.cuda().float(), self.one_hots_U.cuda().float(), self.sample_lens_U.cuda().float()

        # print('>>>>> ------------------------- zero_shot_rare_first -----------------------<<<<<<<<<<<<<<<<<<<<<<<')
        # self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.load_cache_model_120(file1=file1, feature='hum_obj',class_nums=self.class_nums, num_shot=num_shot, filtered_idx=self.filtered_idx)
        # self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO  = self.cache_model_HO.cuda().float(), self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()
        
        # print('[INFO]: cache model shape:', self.cache_model_HO.shape[0])
        self.use_type = 'crop'

        self.finetune_adapter = False
        if self.finetune_adapter:
            raise NotImplementedError

        self.evaluate_type = 'detr' # gt detr
        self.post_process = args.post_process
        print('[INFO]: post_process:', self.post_process)
        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]

    
        self.obj_to_no_interaction = torch.as_tensor([169,  23,  75, 159,   9,  64, 193, 575,  45, 566, 329, 505, 417, 246,
         30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
         91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        # calculate invalid pairs acc
        self.r1_nopair = 0
        self.count_nopair = 0

        self.r1_pair = 0
        self.count_pair = 0

    def old_prompt2new_prompt(self, prompt):
        '''
        param prompt: str, e.g. 'a photo of a person holding a bicycle'
        '''
        
        new_prefix = 'interaction between human and an object, '
        ## remove old prefix
        prompt = ' '.join(prompt.split()[3:])
        return new_prefix + prompt

    def get_multi_prompts(self, hico_labels):   ## xx
        templates = ['itap of {}', '“a bad photo of {}', 'a photo of {}', 'there is {} in the video game', 'art of {}', 'the picture describes {}']
        hico_texts = [hico_text_label[id].split(' ')[3:] for id in hico_text_label.keys()]
        all_texts_input = []
        for temp in templates:
            texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
            all_texts_input.append(texts_input)
        all_texts_input = torch.stack(all_texts_input,dim=0)
        return all_texts_input
    
    def get_attention_feature(self, query_feat, human_feat, object_feat, ftype='patch'):  ## xxx
        device = torch.device('cuda')
        
        human_feat = human_feat.flatten(2).to(device)
        object_feat = object_feat.flatten(2).to(device)
        key_feat = torch.cat([human_feat,object_feat],dim=-1)

        query_feat = query_feat.flatten(2).transpose(1,2).to(device)
        
        global_feat = query_feat.mean(1)
        # key_feat = key_feat/key_feat.norm(dim=1, keepdim=True)
        # query_feat = query_feat/query_feat.norm(dim=-1, keepdim=True)
        weight_matrix = torch.bmm(query_feat, key_feat)
        weight_matrix = weight_matrix.float().softmax(-1)
        weight_query_feat = torch.bmm(weight_matrix, key_feat.transpose(1, 2).float()).mean(1)
        query_feat = weight_query_feat.float()
        return query_feat.cpu()

    def get_rare_feats(self, hum_embddings, obj_embeddings, union_embeddings, feat_type: str, num_shot: int):
        '''
        rare: #anno <= num_shot
        return List[rare features] --cat--> tensor: NxC
        '''
        rare_feats = []
        for hum_emb, obj_emb, uni_emb in zip(hum_embddings, obj_embeddings, union_embeddings,):
            if len(hum_emb) <= num_shot:
                if feat_type == 'hum_obj':
                    rare_feats.append(torch.cat((torch.as_tensor(hum_emb), torch.as_tensor(obj_emb)), dim=-1))
                elif feat_type == 'hum_obj_uni':
                    rare_feats.append(torch.cat((torch.as_tensor(hum_emb), torch.as_tensor(obj_emb), torch.as_tensor(uni_emb)), dim=-1))
        rare_feats = torch.cat(rare_feats, dim=0)
        return rare_feats

    def load_cache_model_120(self,file1, feature='hum_obj',class_nums=117, num_shot=10, filtered_idx=[]):  ## √
        
        annotation = pickle.load(open(file1,'rb'))
        # if category == 'verb':
        categories = class_nums
        union_embeddings = [[] for i in range(categories)]
        obj_embeddings = [[] for i in range(categories)]
        hum_embeddings = [[] for i in range(categories)]
        filenames = list(annotation.keys())
        # verbs_iou = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
        # hois_iou = [[] for i in range(len(hois))]
        # filenames = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
        # each_filenames = [[] for i in range(categories)]
        # sample_indexes = [[] for i in range(categories)]
        for file_n in filenames:
            anno = annotation[file_n]
            if categories == 117: verbs = anno['verbs']
            else: verbs = anno['hois']
            
            union_features = anno['union_features']
            object_features = anno['object_features']            
            huamn_features = anno['huamn_features']
  
            if len(verbs) == 0:
                print(file_n)
            for i, v in enumerate(verbs):
                if v in filtered_idx:
                    continue
                union_embeddings[v].append(union_features[i] / np.linalg.norm(union_features[i]))
                obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i]))
                hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))

        assert class_nums == 600
        for v in range(class_nums):
            if v in filtered_idx:
                assert len(obj_embeddings[v]) == 0
                inter_hum_indices = (torch.as_tensor(self.HOI_IDX_TO_ACT_IDX) == self.HOI_IDX_TO_ACT_IDX[v]).nonzero()
                inter_hum_indices = [i for i in inter_hum_indices if i not in filtered_idx]
                hum_emb = []
                for i in inter_hum_indices:
                    hum_emb.extend(hum_embeddings[i])

                inter_obj_indices = (torch.as_tensor(self.HOI_IDX_TO_OBJ_IDX) == self.HOI_IDX_TO_OBJ_IDX[v]).nonzero()
                inter_obj_indices = [i for i in inter_obj_indices if i not in filtered_idx]
                obj_emb = []
                for i in inter_obj_indices:
                    obj_emb.extend(obj_embeddings[i])
                
                num_to_reserve = min(100, len(hum_emb), len(obj_emb))
                hum_embeddings[v] = random.sample(hum_emb, num_to_reserve)
                obj_embeddings[v] = random.sample(obj_emb, num_to_reserve)

        cache_models_lst, each_lens_lst = [], []
        if feature == 'hum_obj':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings)):
                hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                new_embeddings = torch.cat([hum_emb, obj_emb], dim=-1)
                new_embeddings = new_embeddings.cuda().float()

                num_to_select = min(hum_emb.shape[0], num_shot)

                topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                new_embeddings = new_embeddings[topk_idx]
                
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
        else:
            raise NotImplementedError
        
        cache_models = torch.cat(cache_models_lst, dim=0) ## todo 
        each_lens = torch.as_tensor(each_lens_lst)  ## 32, 32, 13, 27, 32, 

        cumsum_sample_lens = torch.cumsum(each_lens, dim=-1) 
        one_hots = torch.zeros(cumsum_sample_lens[-1], class_nums)
        for z in range(class_nums):
            if z == 0:
                one_hots[0:cumsum_sample_lens[z], z] = 1
            else:
                one_hots[cumsum_sample_lens[z-1]:cumsum_sample_lens[z], z] = 1
        return cache_models, one_hots, each_lens
        

    def load_cache_model(self,file1, feature='union',class_nums=117, num_shot=10, use_multi_hot=False, label_choice='random', num_anno=None):  ## √
        annotation = pickle.load(open(file1,'rb'))
        categories = class_nums
        if self.dic_key == 'verb':
            num_dic_key = 24 if categories == 24 else 117
        elif self.dic_key == 'object':
            num_dic_key = 80
        else:
            num_dic_key = 600 
        
        union_embeddings = [[] for i in range(num_dic_key)]
        obj_embeddings = [[] for i in range(num_dic_key)]
        hum_embeddings = [[] for i in range(num_dic_key)]
        real_verbs = [[] for i in range(num_dic_key)]
        filenames = list(annotation.keys())
        verbs_iou = [[] for i in range(num_dic_key)] # contain 600hois or 117 verbs
        # hois_iou = [[] for i in range(len(hois))]
        # filenames = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
        each_filenames = [[] for i in range(num_dic_key)]
        for file_n in filenames:
            anno = annotation[file_n]
            # dict_keys (['boxes_h', 'boxes_o', 'verbs', 'hois', 'union_boxes', 'union_features', 'huamn_features''object_features''objects', 'global_feature'])
            if self.dic_key == 'verb': verbs = anno['verbs']
            elif self.dic_key == 'object': verbs = anno['objects']
            else: verbs = anno['hois']

            if verbs.shape == ():
                verbs = np.array([verbs])
            num_ho_pair = len(anno['boxes_h'])
            anno['real_verbs'] = np.zeros(shape=(num_ho_pair, categories))
            for i in range(num_ho_pair):
                anno['real_verbs'][i][verbs[i]] = 1
            
            if use_multi_hot:
                co_occur_lst = []
                tgt_idx = []
                boxes_h_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_h']))
                boxes_o_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_o']), torch.as_tensor(anno['boxes_o']))
                for i in range(num_ho_pair):
                    idx_h = boxes_h_iou[i] > 0.5
                    idx_o = torch.logical_and(boxes_o_iou[i] > 0.5, torch.as_tensor(anno['objects']) == anno['objects'][i])
                    idx_ho = torch.logical_and(idx_h, idx_o)
                    anno['real_verbs'][i] = torch.sum(torch.as_tensor(anno['real_verbs'])[idx_ho], dim=0)
                    
                    cur_cooccur_idxs = idx_ho.nonzero().squeeze(-1)
                    for item in co_occur_lst:
                        if (len(item) == len(cur_cooccur_idxs)) and (item == cur_cooccur_idxs).all():
                            break
                    else:
                        co_occur_lst.append(cur_cooccur_idxs)
                anno['real_verbs'][anno['real_verbs']>1] = 1
                for item in co_occur_lst:
                    tgt_idx.append( item[random.randint( 0, len(item) -1 )].item() )

            ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
            if len(verbs) == 0:
                print(file_n)
            for i, v in enumerate(verbs):
                if use_multi_hot and self.rm_duplicate_feat:
                    if i not in tgt_idx:
                        continue
                union_embeddings[v].append(anno['union_features'][i] / np.linalg.norm(anno['union_features'][i]))
                obj_embeddings[v].append(anno['object_features'][i] / np.linalg.norm(anno['object_features'][i]))
                hum_embeddings[v].append(anno['huamn_features'][i] / np.linalg.norm(anno['huamn_features'][i]))
                each_filenames[v].append(file_n)
                real_verbs[v].append(anno['real_verbs'][i])
                # add iou
                verbs_iou[v].append(ious[i])
        
        ## re-implement cachemodel construction
        cache_models_lst, each_lens_lst = [], []
        real_verbs_lst = []
        if feature == 'hum_obj':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = torch.cat([hum_emb, obj_emb], dim=-1)
                new_embeddings = new_embeddings.cuda().float()
                
                if self.sample_choice == 'uniform':
                    num_to_select = min(hum_emb.shape[0], num_shot)
                elif self.sample_choice == 'origin':
                    num_to_select = max(hum_emb.shape[0] // 16, 1)

                if num_to_select < hum_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        elif feature == 'hum_obj_uni':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                uni_emb = torch.as_tensor(np.array(embeddings)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = torch.cat([hum_emb, obj_emb, uni_emb], dim=-1)
                new_embeddings = new_embeddings.cuda().float()

                if self.sample_choice == 'uniform':
                    num_to_select = min(hum_emb.shape[0], num_shot)
                elif self.sample_choice == 'origin':
                    num_to_select = max(hum_emb.shape[0] // 16, 1)

                if num_to_select < hum_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        elif feature == 'uni':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                uni_emb = torch.as_tensor(np.array(embeddings)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = uni_emb
                new_embeddings = new_embeddings.cuda().float()

                if self.sample_choice == 'uniform':
                    num_to_select = min(uni_emb.shape[0], num_shot)
                elif self.sample_choice == 'origin':
                    num_to_select = max(uni_emb.shape[0] // 16, 1)

                if num_to_select < uni_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v) 
        else:
            raise NotImplementedError
        
        cache_models = torch.cat(cache_models_lst, dim=0)
        labels = torch.cat(real_verbs_lst, dim=0)
        return cache_models, labels, torch.sum(labels, dim=0)

        
    def get_clip_feature(self,image):  ## xxx
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        # x = self.clip_model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        return local_feat
    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:  ### √
        
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
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
        ## logits.shape == prior_h.shape == prior_o.shape,
        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
        
    def compute_text_embeddings(self):  ### xxx
        text_embeddings = self.clip_head.text_encoder(self.texts)
        return text_embeddings

    def compute_roi_embeddings_targets(self, features: OrderedDict, image_shapes: Tensor, targets_region_props: List[dict], return_type='crop'): ### √
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        all_logits = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_boxes = []
        for i, targets in enumerate(targets_region_props):
            local_features = features[i]
            gt_bx_h = (box_ops.box_cxcywh_to_xyxy(targets['boxes_h']) * scale_fct[i][None,:])
            gt_bx_o = (box_ops.box_cxcywh_to_xyxy(targets['boxes_o']) * scale_fct[i][None,:])
            verbs = targets['labels']
            # hois = targets['hoi']
            filename = targets['filename']
            objects_label = targets['object']
            lt = torch.min(gt_bx_h[..., :2], gt_bx_o[..., :2]) # left point
            rb = torch.max(gt_bx_h[..., 2:], gt_bx_o[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            if return_type == 'roi':
                union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(1, 1),spatial_scale=1 / 14.0,aligned=True).flatten(2).mean(-1)
                huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(1, 1),spatial_scale=1 / 14.0,aligned=True).flatten(2).mean(-1)
                object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(1, 1),spatial_scale=1 / 14.0,aligned=True).flatten(2).mean(-1)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                huamn_features = huamn_features / huamn_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                logits = union_features @ self.text_embedding.t()
                logits_cache = ((torch.cat((huamn_features, object_features, union_features), dim=1) @ self.cache_models.t()) @ self.one_hots) / self.sample_lens
                logits = logits + logits_cache
            elif return_type == 'crop':  # to do -> cache model 
                lens = local_features.shape[0]
                huamn_features = local_features[:lens//3,:] # human features
                object_features = local_features[lens//3:lens//3*2,:] # object features
                union_features = local_features[lens//3*2:,:]  # union features
                
                huamn_features = huamn_features / huamn_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                
                if self.feature == 'hum_obj_uni':
                    f_vis = torch.cat((huamn_features, object_features, union_features), dim=-1)
                elif self.feature == 'uni':
                    f_vis = union_features
                elif self.feature == 'hum_obj':
                    f_vis = torch.cat((huamn_features, object_features), dim=-1)
                else:
                    print(f"[ERROR]: feature_type {self.feature} not implemented yet")
                if self.branch == 'vis_only':
                    logits = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                elif self.branch == 'text_only':
                    logits = union_features @ self.text_embedding.t()
                elif self.branch == 'vis+text':
                    logits_v = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                    logits_v /= len(self.feature.split('_'))
                    logits_t = union_features @ self.text_embedding.t()
                    logits = self.alpha * logits_v + logits_t

            else:
                print('please input the correct return type: roi or crop')
                sys.exit()

            all_boxes.append(torch.cat([gt_bx_h,gt_bx_o],dim=0))
            keep = torch.arange(len(gt_bx_h)*2).to(local_features.device)
            boxes_h_collated.append(keep[:len(gt_bx_h)])
            boxes_o_collated.append(keep[len(gt_bx_h):])
            object_class_collated.append(objects_label)
            scores = torch.ones(len(keep)).to(local_features.device)
            
            prior_collated.append(self.compute_prior_scores(
                keep[:len(gt_bx_h)], keep[:len(gt_bx_o)], scores, objects_label)
            )
            all_logits.append(logits.float())
        return all_logits,prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated ,all_boxes


    def compute_roi_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]): ### xx
        pass

    def compute_crop_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict], targets_region_props: List[dict], return_type='crop'): ### √
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        all_logits = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_boxes = []
        device = features.device
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx].float()
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels'] ## instance class(0~80)
            is_human = labels == self.human_idx
            
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; 
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
            # x_keep,y_keep都是boxes的indices
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
            union_boxes = torch.cat([lt,rb],dim=-1)

            ## box_cxcywh_to_xyxy
            gt_bx_h = self.recover_boxes(targets_region_props[b_idx]['boxes_h'], targets_region_props[b_idx]['size']) 
            gt_bx_o = self.recover_boxes(targets_region_props[b_idx]['boxes_o'], targets_region_props[b_idx]['size'])
            
            x, y = torch.nonzero(torch.min(
                box_iou(sub_boxes, gt_bx_h),
                box_iou(obj_boxes, gt_bx_o)
                ) >= self.fg_iou_thresh).unbind(1)
            
            if len(x) != 0:
                x = torch.as_tensor(list(set(x.cpu().numpy()))).to(x.device)
            if len(x) == 0: 
                # print(x,y)
                # self.count += 1
                pass

            ## 没有和gt bbox匹配上的bbox的index
            no_pair_x =  list(set(np.arange(len(sub_boxes)).tolist()) - set(x.cpu().numpy().tolist()))
            # x_keep = x_keep[x]
            # y_keep = y_keep[x]
            
            lens = local_features.shape[0]
            feat1 = local_features[:lens//3,:] # human features
            feat2 = local_features[lens//3:lens//3*2,:] # object features
            feat3 = local_features[lens//3*2:,:]  # union features
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            feat3 = feat3 / feat3.norm(dim=-1, keepdim=True)

            if self.logits_type == 'HO+U+T':
                logits_HO = (torch.cat((feat1, feat2), dim=-1) @ self.cache_model_HO.t()) @ self.one_hots_HO / self.sample_lens_HO
                logits_HO /= 2
                logits_U = (feat3 @ self.cache_model_U.t()) @ self.one_hots_U / self.sample_lens_U
                logits_text = feat3 @ self.text_embedding.t()
                logits = (self.gamma_HO * logits_HO + self.gamma_U * logits_U + logits_text) / (self.gamma_HO+self.gamma_U+1)
            elif self.logits_type == "HO+U":
                logits_HO = (torch.cat((feat1, feat2), dim=-1) @ self.cache_model_HO.t()) @ self.one_hots_HO / self.sample_lens_HO
                logits_HO /= 2
                logits_U = (feat3 @ self.cache_model_U.t()) @ self.one_hots_U / self.sample_lens_U
                logits = (self.gamma_HO * logits_HO + self.gamma_U * logits_U) / (self.gamma_HO+self.gamma_U)
            elif self.logits_type == 'U+T':
                logits_U = (feat3 @ self.cache_model_U.t()) @ self.one_hots_U / self.sample_lens_U
                logits_text = feat3 @ self.text_embedding.t()
                logits = (self.gamma_U * logits_U + logits_text)/ (self.gamma_U+1)
            elif self.logits_type == 'HO+T':
                logits_HO = (torch.cat((feat1, feat2), dim=-1) @ self.cache_model_HO.t()) @ self.one_hots_HO / self.sample_lens_HO
                logits_HO /= 2
                logits_text = feat3 @ self.text_embedding.t()
                logits = (self.gamma_HO * logits_HO + logits_text)/ (self.gamma_HO+1)
            elif self.logits_type == 'T':
                logits_text = feat3 @ self.text_embedding.t()
                logits = logits_text
            elif self.logits_type == 'HO':
                logits_HO = (torch.cat((feat1, feat2), dim=-1) @ self.cache_model_HO.t()) @ self.one_hots_HO / self.sample_lens_HO
                logits_HO /= 2
                logits = logits_HO
            elif self.logits_type == 'U':
                logits_U = (feat3 @ self.cache_model_U.t()) @ self.one_hots_U / self.sample_lens_U
                logits = logits_U

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])  ## labels of detected instances
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            if self.post_process:
                mask = torch.zeros_like(logits)
                for i, obj_l in enumerate(labels[y_keep]):
                    possible_targets = torch.as_tensor(self.object_class_to_target_class[obj_l])
                    if len(possible_targets) == 0:
                        continue
                    mask[i][possible_targets] = 1
                logits = logits.masked_fill(mask == False, float('-inf'))
                logits = (logits/self.temperature).softmax(-1)

            all_logits.append(logits)
            
        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated
        
    
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training 
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

    def prepare_region_proposals(self, results): ## √ detr extracts the human-object pairs
        region_props = []
        for res in results:
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)

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
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
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
    
    def get_region_proposals(self, results): ##  √√√
        region_props = []
        for res in results:
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
                keep_h = hum

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
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
        
    def get_targets_pairs(self, targets): ### xxxxxxxxxx
        region_targets = {}
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        for tar in targets:
            
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
    
    def get_prior(self, region_props,image_size): ##  for adapter module training
        
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
                
                print(n_h,n)
                # sys.exit()
            
            object_embs = self.object_embedding[labels]
            priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            priors[b_idx,:n,5:] = object_embs
            mask[b_idx,:n] = False
        
        priors = self.priors(priors)
        return (priors, mask)

    def get_paired_bbox_proposals(self, results, image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        bx = results['boxes']
        sc = results['scores']
        lb = results['labels'] ## object-category labels(0~80)

        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = bx.device
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)

        return sub_boxes, obj_boxes, union_boxes
    
    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = PIL.Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = PIL.Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
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
        if self._load_features:
            image_sizes = torch.stack([t['size'] for t in targets])
            region_props = self.get_region_proposals(targets)
            cls_feature = images[0]
            logits, prior, bh, bo, objects = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
            boxes = [r['boxes'] for r in region_props]   
        else:
            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            
            image_sizes = torch.as_tensor([
                im.size()[-2:] for im in images
            ], device=images[0].device)
            # get detr results
            region_props = self.get_region_proposals(targets) ## List[Dict], dict_keys(['boxes','scores', 'labels', 'hidden_states', 'mask'])
            # feat_local_old = self.clip_model.encode_image(images[0]) ## images[0]: Nx3x224x224, N: hum,hum,hum,obj,obj,obj,uni,uni,uni
            feat_local_old = forward_chunks(self.clip_model.encode_image, images[0]) ## batch-size must be 1
            # feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 14, 14).float()
            cls_feature = feat_local_old[:,0,:]

            if self.evaluate_type == 'gt': 
                if self.use_type == 'crop':
                    logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(cls_feature.unsqueeze(0), image_sizes, targets)
                else: 
                    logits, prior, bh, bo, objects = self.compute_roi_embeddings(feat_local, image_sizes, region_props)
            elif self.evaluate_type == 'detr': ## during testing
                logits, prior, bh, bo, objects = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
                boxes = [r['boxes'] for r in region_props]   
        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

def build_detector(args, class_corr, num_anno):
    if args.d_detr:
        detr, _, postprocessors = build_model_d_detr(args)
    else:
        detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    
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
    detector = UPT( args,
        detr, postprocessors['bbox'], clip_head, args.clip_dir_vit,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        num_anno = num_anno,
    )
    return detector
