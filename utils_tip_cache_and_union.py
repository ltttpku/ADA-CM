"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import detr.datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image

def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
        
        # images_clip.append(im_clip)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, clip_model_name):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        assert clip_model_name in ['ViT-L/14@336px', 'ViT-B/16']
        self.clip_model_name = clip_model_name
        if self.clip_model_name == 'ViT-B/16':
            self.clip_input_resolution = 224
        elif self.clip_model_name == 'ViT-L/14@336px':
            self.clip_input_resolution = 336

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            # if partition == 'train2015':
            #     self.anno_bbox = pickle.load(open('hico_train_bbox_max25.p','rb'))
            # else:
            #     self.anno_bbox = pickle.load(open('hico_test_bbox.p','rb'))
            # pdb.set_trace()
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # add clip normalization 
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        normalize_clip = T.Compose([
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        normalize_clip_1 = T.ToTensor()
        normalize_clip_2 = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = [T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ]))
                    ]),
        normalize, normalize_clip,
        T.Compose([
                T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ])
        ]
        else:   
            self.transforms = [T.Compose([
                T.RandomResize([800], max_size=1333),
            ]),
            normalize, normalize_clip,
            T.Compose([
                 T.IResize([self.clip_input_resolution,self.clip_input_resolution])
            ]),
            normalize_clip_1,
            normalize_clip_2

            ]

        self.partition = partition
        self.name = name
        self.count=0
        
        device = "cuda"
        # _, self.process = clip.load('ViT-B/16', device=device)
        _, self.process = clip.load(self.clip_model_name, device=device)

    def __len__(self):
        return len(self.dataset)

    ###  use gt box and class
    def __getitem__(self, i):
        (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w]) 

        if self.name == 'hicodet': # dict_keys(['boxes_h', 'boxes_o', 'hoi', 'object', 'verb', 'orig_size'])
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
            target['hoi'] = torch.as_tensor((np.array(np.array(self.dataset.object_n_verb_to_interaction)[target['object'], target['actions']])).astype(np.int64))
        # test

        lt = torch.min(target['boxes_h'][...,:2],target['boxes_o'][...,:2])
        rb = torch.max(target['boxes_h'][..., 2:], target['boxes_o'][..., 2:])
        crop_size=torch.cat([lt,rb],dim=-1).numpy()
        crop_size_object = target['boxes_o'].numpy()
        crop_size_human = target['boxes_h'].numpy()
        all_images = []
        all_objects = []
        all_human = []

        # background_pixels = (int(0.48145466*255),int(0.4578275*255), int(0.40821073*255))  
        background_pixels = (0,0,0)
        for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
            new_img = image.crop(crop_s)
            new_img = self.expand2square(new_img,background_pixels) #
            all_images.append(self.process(new_img)) 
            new_img = image.crop(crop_s_o)
            new_img = self.expand2square(new_img,background_pixels) #
            all_objects.append(self.process(new_img))
            new_img = image.crop(crop_s_h)
            new_img = self.expand2square(new_img,background_pixels) #
            all_human.append(self.process(new_img))
        
        all_images = torch.stack(all_images)
        all_images_object = torch.stack(all_objects)
        all_images_human = torch.stack(all_human)

        all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
        # pdb.set_trace()
        image_0, target_0 = self.transforms[3](image, target)
        image_clip, target = self.transforms[2](image_0, target_0)
        if image_0.size[-1] >self.clip_input_resolution  or image_0.size[-2] >self.clip_input_resolution  :print(image_0.size)
        target['filename'] = filename
        
        # mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
        # for i in range(len(target['ex_bbox'])):
        #     t = target['ex_bbox'][i].clamp(0,224).int()
        #     mask[i, t[1]:t[3], t[0]:t[2]] = 1
        # # pdb.set_trace()
        # assert mask.shape[0] != 0
        # mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
        # target['ex_mask'] = mask

        all_images = torch.cat([image_clip.unsqueeze(0), all_images],dim=0)
        return all_images,target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        # for res in results:
        # pdb.set_trace()
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            # keep_h = keep[keep_h]
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            # keep_o = keep[keep_o]
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
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
            # boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
            # continue

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
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
      
        # region_props.append(dict(
        #     boxes=bx[keep],
        #     scores=sc[keep],
        #     labels=lb[keep],
        #     hidden_states=hs[keep],
        #     mask = ms[keep]
        # ))

        # return sub_boxes.int(), obj_boxes.int(), union_boxes.int()
        return sub_boxes, obj_boxes, union_boxes

    def get_union_mask(self, bbox, image_size):
        n = len(bbox)
        masks = torch.zeros
        pass
class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        # self.test_loader = kwargs['test_loader']
        # self.anno_interaction = kwargs['anno_interaction']
        # self.cache_dir = kwargs['cache_dir']
    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    @torch.no_grad()
    def test_hico(self, dataloader, datasetname='hicodet'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        # associate = BoxPairAssociation(min_iou=0.5)
        # conversion = torch.from_numpy(np.asarray(
        #     dataset.object_n_verb_to_interaction, dtype=float
        # ))
        # pdb.set_trace()
        # interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)
        
        # meter = DetectionAPMeter(
        #     600, nproc=1,
        #     num_gt=dataset.anno_interaction,
        #     algorithm='11P'
        # )
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # pdb.set_trace()
            outputs = net(inputs,batch[1]) 
            continue

        with open(f'{datasetname}_pkl_files/{datasetname}_union_embeddings_cachemodel_crop_padding_zeros_vit336_TF.p',"wb") as f:
            pickle.dump(net.module.dicts,f)
        exit(0)
        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
