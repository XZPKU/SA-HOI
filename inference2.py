"""
Visualise detected human-object interactions in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import pocket
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import os.path as osp
import PIL.Image as Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('/scratch/yangdejie/xz/ReVersion/')
from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
from hoi_configs.human import human_idx
# from utils import DataFactory
from utils_tip_cache_and_union_finetune import custom_collate, CustomisedDLE, DataFactory

# from upt import build_detector
from upt_tip_cache_model_free_finetune_distill3 import build_detector
import pdb
import random
from pocket.ops import relocate_to_cpu, relocate_to_cuda
warnings.filterwarnings("ignore")

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i+1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(image, output, actions, action=None, thresh=0.2, save_filename=None, failure=False):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct
    # Find the number of human and object instances
    nh = len(output['pairing'][0].unique()); no = len(boxes)

    scores = output['scores']
    objects = output['objects']
    pred = output['labels']
    
    # Visualise detected human-object pairs with attached scores
    # pdb.set_trace()
    unique_actions = torch.unique(pred)
    
    if action is not None:
        plt.cla()
        if failure:
            keep = torch.nonzero(torch.logical_and(scores < thresh, pred == action)).squeeze(1)
        else:
            keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        bx_h, bx_o = boxes[output['pairing']].unbind(0)
        pocket.utils.draw_box_pairs(image, bx_h[keep], bx_o[keep], width=5)
        plt.imshow(image)
        plt.axis('off')
        # pdb.set_trace()
        if len(keep) == 0: return 
        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()
        # plt.show()
        plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
        # plt.savefig(save_filename)
        plt.cla()
        return

    pairing = output['pairing']
    # coop_attn = output['attn_maps'][0]
    # comp_attn = output['attn_maps'][1]

    # Visualise attention from the cooperative layer
    # for i, attn_1 in enumerate(coop_attn):
    #     fig, axe = plt.subplots(2, 4)
    #     fig.suptitle(f"Attention in coop. layer {i}")
    #     axe = np.concatenate(axe)
    #     ticks = list(range(attn_1[0].shape[0]))
    #     labels = [v + 1 for v in ticks]
    #     for ax, attn in zip(axe, attn_1):
    #         im = ax.imshow(attn.squeeze().T, vmin=0, vmax=1)
    #         divider = make_axes_locatable(ax)
    #         ax.set_xticks(ticks)
    #         ax.set_xticklabels(labels)
    #         ax.set_yticks(ticks)
    #         ax.set_yticklabels(labels)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax)

    # x, y = torch.meshgrid(torch.arange(nh), torch.arange(no))
    # x, y = torch.nonzero(x != y).unbind(1)
    # pairs = [str((i.item() + 1, j.item() + 1)) for i, j in zip(x, y)]

    # Visualise attention from the competitive layer
    # fig, axe = plt.subplots(2, 4)
    # fig.suptitle("Attention in comp. layer")
    # axe = np.concatenate(axe)
    # ticks = list(range(len(pairs)))
    # for ax, attn in zip(axe, comp_attn):
    #     im = ax.imshow(attn, vmin=0, vmax=1)
    #     divider = make_axes_locatable(ax)
    #     ax.set_xticks(ticks)
    #     ax.set_xticklabels(pairs, rotation=45)
    #     ax.set_yticks(ticks)
    #     ax.set_yticklabels(pairs)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im, cax=cax)

    # Print predicted actions and corresponding scores
    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {actions[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] + 1
            print(
                f"({idxh.item():<2}, {idxo.item():<2}),",
                f"score: {scores[idx]:.4f}, object: {OBJECTS[objects[idx]]}."
            )
    
    # Draw the bounding boxes
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    pdb.set_trace()
    ax = plt.gca()
    draw_boxes(ax, boxes)
    # plt.show()
    plt.savefig('visualizations/test.png')


def postprocess(output, actions):

    h, w = output['size']

    boxes = output['boxes'] 
    nh = len(output['pairing'][0].unique()); no = len(boxes)

    scores = output['scores']
    objects = output['objects']
    pred = output['labels']
    

    unique_actions = torch.unique(pred)

    pairing = output['pairing']
    unique_actions = torch.unique(pred)
    hois = []
    for verb in unique_actions:
        #print(f"\n=> Action: {actions[verb]}")
        action_name = actions[verb]
        #hoi_verb = []
       
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            idxh, idxo = pairing[:, idx] 
            hoi_verb_idx={}
            hoi_verb_idx['human'] = idxh.item()
            hoi_verb_idx['object'] = idxo.item()
            hoi_verb_idx['human_box'] = boxes[idxh].tolist()
            hoi_verb_idx['object_box'] = boxes[idxo].tolist()
            hoi_verb_idx['score'] = scores[idx].item()
            hoi_verb_idx['object_name'] = OBJECTS[objects[idx]]
            hoi_verb_idx['verb'] = action_name
            hois.append(hoi_verb_idx)
            # print(
            #     f"({idxh.item():<2}, {idxo.item():<2}),",
            #     f"score: {scores[idx]:.4f}, object: {OBJECTS[objects[idx]]}."
            # )
    return hois   

import detr.datasets.transforms_clip as T
from PIL import Image
def transform(image):
    transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
    clip_transforms = T.Compose([
                T.IResize([224,224]),
            ])
    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image,_ = transforms(image,None)
    image_clip,_ = clip_transforms(image,None)  
    image, _ = normalize(image, None)
    image_clip, _ = normalize(image_clip,None)
    image = image.to('cuda')
    image_clip = image_clip.to('cuda')
    return (image,image_clip), torch.zeros((3,224,224))
import json
def check(hoid_result,gt_hoi):
    correct_hoi_score = []
    for hoi_instance in hoid_result:
        verb = hoi_instance['verb']
        verb_ing = verb  + 'ing'
        verb_ing_2 = verb[:-1] +'ing'
        obj = hoi_instance['object_name']
        if obj in gt_hoi and (verb in gt_hoi or verb_ing in gt_hoi or verb_ing_2 in gt_hoi):
            correct_hoi_score.append(hoi_instance['score'])
    if len(correct_hoi_score)==0:
        return 0
    return sum(correct_hoi_score)

def obj_check(hoid_result,gt_hoi):
    for hoi_instance in hoid_result:
        obj = hoi_instance['object_name']
        if obj in gt_hoi:
            return 1
    return 0
@torch.no_grad()
def main(args):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=0
    )

    random.seed(1234)
    args.clip_model_name = args.clip_dir_vit.split('/')[-1].split('.')[0]
    if args.clip_model_name == 'ViT-B-16':
        args.clip_model_name = 'ViT-B/16' 
    elif args.clip_model_name == 'ViT-L-14-336px':
        args.clip_model_name = 'ViT-L/14@336px'
    args.human_idx = 0 
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root, clip_model_name=args.clip_model_name)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    actions = dataset.dataset.verbs if args.dataset == 'hicodet' else \
        dataset.dataset.actions

    # actions = dataset.dataset.interactions
    # object_to_target = dataset.dataset.object_to_verb
    object_n_verb_to_interaction = dataset.dataset.object_n_verb_to_interaction
    num_anno = torch.as_tensor(dataset.dataset.anno_action)
    upt = build_detector(args, conversion, object_n_verb_to_interaction=object_n_verb_to_interaction, clip_model_path=args.clip_dir_vit, num_anno=num_anno, verb2interaction=None)
    upt = upt.cuda()
    upt.eval()

    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Start from a randomly initialised model")
    hoi_root_pth = osp.join(args.hoi_data_root,f'{args.hoi_type}/{args.syn_model}')
    if args.hoi_type=='animal':
        hoi_config = animal_idx
    elif args.hoi_type == 'connect':
        hoi_config = connect_idx
    else:
        hoi_config = human_idx
    hoi_scores = []
    obj_scores = []
    for hoi_key in hoi_config.keys():
        hoi_root=  os.path.join(hoi_root_pth,str(hoi_key))
        hoid_result = {}
        image_names = os.listdir(hoi_root)
        gt_hoi_category = hoi_config[hoi_key]
        for name in image_names:
            if not name.endswith('.png'):
                continue 
            hoi_full_pth = os.path.join(hoi_root,name)
            image = Image.open(hoi_full_pth)
            image_tensor, _ = transform(image)
            output = upt([image_tensor])
            if output is None:
                continue
            hois = postprocess(output[0],actions)
            hoid_result[hoi_full_pth] = hois
            score = check(hoid_result[hoi_full_pth],gt_hoi_category)
            object_fidelity = obj_check(hoid_result[hoi_full_pth],gt_hoi_category)
            hoi_scores.append(score)
            obj_scores.append(object_fidelity)
        with open(osp.join(args.hoi_data_root,f'{args.hoi_type}/{args.syn_model}/{hoi_key}/hoid.json'),'w') as file:
            json.dump(hoid_result,file)
        print(f'finish hoid for class {hoi_key} for {args.hoi_type} for {args.syn_model}.\n')
    hoi_fidelty = [item for item in hoi_scores if item > 0.1]
    hoif = sum(hoi_fidelty)/len(hoi_fidelty)
    print(f'hoif for {args.syn_model} on {args.hoi_type} is {hoif}.')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)
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
    parser.add_argument('--partition', default='test2015', type=str)
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
    # parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    # parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    # parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    # parser.add_argument('--clip_visual_width', default=64, type=int)
    # parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    # parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    # parser.add_argument('--clip_text_context_length', default=13, type=int)

    #### add CLIP vision transformer
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)

    ### ViT-L/14@336px START: emb_dim: 768 
    # >>> vision_width: 1024,  vision_patch_size(conv's kernel-size&&stride-size): 14,
    # >>> vision_layers(#layers in vision-transformer): 24 ,  image_resolution:336;
    # >>> transformer_width:768, transformer_layers: 12, transformer_heads:12
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)

    # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-L/14@336px----END----
    
    ### ViT-B-16 START
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)

    # # parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    # ---END----ViT-B-16-----END-----
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int) # 13 -77
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_consistloss', action='store_true')
    parser.add_argument('--use_multi_hot', action='store_true')
    parser.add_argument('--label_learning', action='store_true')
    parser.add_argument('--label_choice', default='random', choices=['random', 'single_first', 'multi_first', 'single+multi', 'rare_first', 'non_rare_first', 'rare+non_rare'])
    parser.add_argument('--use_mlp_proj', action='store_true')

    parser.add_argument('--adapter_pos', type=str, default='all', choices=['all', 'front', 'end', 'random', 'last'])
    parser.add_argument('--adapter_num_layers', type=int, default=1)
    parser.add_argument('--obj_affordance', action='store_true') ## use affordance embedding of objects
    
    parser.add_argument('--use_mean', action='store_true') # 13 -77
    parser.add_argument('--logits_type', default='HO+U+T', type=str) # 13 -77 # text_add_visual, visual
    parser.add_argument('--num_shot', default='2', type=int) # 13 -77 # text_add_visual, visual
    parser.add_argument('--obj_classifier', action='store_true') # 
    parser.add_argument('--classifier_loss_w', default=1.0, type=float)
    parser.add_argument('--file1', default='./hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p',type=str)
    parser.add_argument('--interactiveness_prob_thres', default=0.1, type=float)
    # parser.add_argument('--feature_type', default='hum_obj_uni', type=str)
    # parser.add_argument('--use_deformable_attn', action='store_true')
    parser.add_argument('--prior_type', type=str, default='cbe', choices=['cbe', 'cb', 'ce', 'be', 'c', 'b', 'e'])
    parser.add_argument('--training_set_ratio', type=float, default=1.0)
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--zs', action='store_true') ## zero-shot
    parser.add_argument('--hyper_lambda', type=float, default=2.8)
    parser.add_argument('--use_weight_pred', action='store_true')

    parser.add_argument('--zs_type', type=str, default='rare_first', choices=['rare_first', 'non_rare_first', 'unseen_verb'])
    parser.add_argument('--domain_transfer', action='store_true') 
    parser.add_argument('--fill_zs_verb_type', type=int, default=0,) # (for init) 0: random; 1: weighted_sum, 
    parser.add_argument('--pseudo_label', action='store_true') 
    parser.add_argument('--tpt', action='store_true') 
    parser.add_argument('--vis_tor', type=float, default=1.0)

    ## prompt learning
    parser.add_argument('--N_CTX', type=int, default=24)  # number of context vectors
    parser.add_argument('--CSC', type=bool, default=False)  # class-specific context
    parser.add_argument('--CTX_INIT', type=str, default='')  # initialization words
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')  # # 'middle' or 'end' or 'front'

    parser.add_argument('--prompt_learning', action='store_true') 
    parser.add_argument('--use_templates', action='store_true') 
    parser.add_argument('--LA', action='store_true')  ## Language Aware
    parser.add_argument('--LA_weight', default=0.6, type=float)  ## Language Aware

    parser.add_argument('--feat_mask_type', type=int, default=0,) # 0: dropout(random mask); 1: 
    parser.add_argument('--num_classes', type=int, default=117,) 
    parser.add_argument('--prior_method', type=int, default=0) ## 0: instance-wise, 1: pair-wise, 2: learnable
    parser.add_argument('--box_proj', type=int, default=0,) ## 0: None; 1: f_u = ROI-feat + MLP(uni-box)

    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
        help="Path to an image file.")
    parser.add_argument('--syn_model', default='sd', type=str)
    parser.add_argument('--syn_num', default=50, type=int)
    parser.add_argument('--hoi_type', default='animal', type=str)
    parser.add_argument('--hoi_data_root',type=str,default='/scratch/yangdejie/xz/ReVersion/clean_hoi')
    args = parser.parse_args()
    args.failure = False
    main(args)
