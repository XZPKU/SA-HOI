from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
from hoi_configs.human import human_idx
import json
from PIL import Image
import os
import torch
import os.path as osp
import torchvision.transforms.functional as F
image_root_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/images/train2015/'
config_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/annotations/trainval_hico.json'
import cv2
import numpy as np
from scipy.spatial import distance
from scipy import ndimage
import argparse
import PIL
from PIL import Image
import numpy as np
from scipy.stats import wasserstein_distance
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
inferencer = MMPoseInferencer(pose2d='human')
hand_inferencer = MMPoseInferencer(pose2d='hand')
animal_inferencer = MMPoseInferencer(pose2d='animal')

def collect_real_json_info_for_single_hoi(json_raw_info):
    body_ratio=[] 
    hand_ratio=[] 
    human_body_keypoint=[] 
    keypoint_score=[]
    keypoint_visible=[]
    human_hand_keypoint=[]
    hand_score_list=[]
    ratio_keypoint = []
    ratio_hand_keypoint = []
    body_distance= []
    hand_distance = []
    for key in json_raw_info:
        values = json_raw_info[key]
        if not values:######## skip bad images##########
            continue
        if 'body_dis' not in values.keys(): ######## skip some img missing distance result####
            continue
        body_ratio.append(values['body_ratio'][0][0])######3only keep first on for convenience for now####
        body_distance.append(values['body_dis'])
        ratio_keypoint.append(values['ratio_keypoint'][0])
        if  'hand_ratio' in values.keys() and values['hand_ratio']>0:
            hand_ratio.append(values['hand_ratio'])
            hand_distance.append(values['hand_dis'])
            ratio_hand_keypoint.append(values['ratio_hand_keypoint'][0])
        human_body_keypoint.append(values['body_pose'])
        keypoint_score.append(values['score'][0])
        keypoint_visible.append(values['visible'][0])
        if 'hand_pose' in values.keys() and not isinstance(values['hand_pose'],int):
            #human_hand_keypoint.append(values['hand_pose'])
            human_hand_keypoint +=values['hand_pose']########## in case one21 hand or 42 two hand#####
        if 'hand_score' in values.keys() and not isinstance(values['hand_score'],int):
            hand_score_list.append(values['hand_score'][0])
    body_ratio = torch.tensor(body_ratio)
    hand_ratio = torch.tensor(hand_ratio)
    body_distance = torch.tensor(body_distance)
    hand_distance = torch.tensor(hand_distance)
    human_body_keypoint = torch.tensor(human_body_keypoint)
    keypoint_score = torch.tensor(keypoint_score)
    keypoint_visible = torch.tensor(keypoint_visible)
    human_hand_keypoint = torch.tensor(human_hand_keypoint)
    hand_score_list = torch.tensor(hand_score_list)
    ratio_keypoint = torch.tensor(ratio_keypoint)
    ratio_hand_keypoint  = torch.tensor(ratio_hand_keypoint)
    all_info = {}
    all_info['body_ratio'] = body_ratio
    all_info['hand_ratio'] = hand_ratio
    all_info['human_body_keypoint'] = human_body_keypoint
    all_info['keypoint_score'] = keypoint_score
    all_info['keypoint_visible'] = keypoint_visible
    all_info['human_hand_keypoint'] = human_hand_keypoint
    all_info['hand_score_list'] = hand_score_list
    all_info['ratio_keypoint'] = ratio_keypoint
    all_info['ratio_hand_keypoint'] = ratio_hand_keypoint
    all_info['hand_distance'] = hand_distance
    all_info['body_distance'] = body_distance
    return all_info
    #return body_ratio,hand_ratio,human_body_keypoint,keypoint_score,keypoint_visible,human_hand_keypoint,hand_score_list
 
def collect_animal_ratio_pose(animal_info):
    
    animal_keypoint  =[]
    animal_score = []
    animal_visible = []
    all_keypoint = []
    all_ratio_keypoint= []
    ratio_keypoint = []
    ratio_animal = []
    for key in animal_info:

        animal_values = animal_info[key]
        animal_keypoint.append(animal_values['keypoint'])
        animal_score.append(animal_values['score'])
        animal_visible.append(animal_values['visible'])
        ratio_animal.append(animal_values['ratio_keypoint'])
        all_keypoint += animal_values['keypoint']
        all_ratio_keypoint +=animal_values['ratio_keypoint']
    all_keypoint = torch.tensor(all_keypoint)
    all_ratio_keypoint = torch.tensor(all_ratio_keypoint)
    return all_ratio_keypoint
    
def collect_joint_info(human_info,animal_info):
    ##################TODO:chage to union bbox size###################
    human_keypoint=[] 
    keypoint_score=[]
    keypoint_visible=[]
    animal_keypoint  =[]
    animal_score = []
    animal_visible = []
    all_keypoint = []
    all_ratio_keypoint= []
    ratio_keypoint = []
    ratio_animal = []
    for key in human_info:
        if key in animal_info.keys():
            values = human_info[key]
            animal_values = animal_info[key]
            if not values or not animal_values:######## skip bad images##########
                continue
            if 'ratio_keypoint' not in animal_values.keys():
                continue
            human_keypoint.append(values['body_pose'])
            keypoint_score.append(values['score'][0])
            keypoint_visible.append(values['visible'][0])
            ratio_keypoint.append(values['ratio_keypoint'])
            animal_keypoint.append(animal_values['keypoint'])
            animal_score.append(animal_values['score'])
            animal_visible.append(animal_values['visible'])
            ratio_animal.append(animal_values['ratio_keypoint'])
            all_keypoint += values['body_pose']
            all_keypoint += animal_values['keypoint']
            # if not len(animal_values['ratio_keypoint']) == 2:
            #     all_ratio_keypoint +=values['ratio_keypoint']
            #     all_ratio_keypoint +=animal_values['ratio_keypoint']
            all_ratio_keypoint.append(animal_values['ratio_keypoint'])
    all_keypoint = torch.tensor(all_keypoint)
    #all_ratio_keypoint = torch.tensor(all_ratio_keypoint)
    all_ratio_keypoint = torch.tensor(all_ratio_keypoint)
    return all_keypoint,all_ratio_keypoint

def collect_json_info_for_single_hoi(json_raw_info,target):
    # pose_ratio = []
    # score = []
    # visible = []
    # body_ratio=[] 
    # hand_ratio=[] 
    # human_body_keypoint=[] 
    # keypoint_score=[]
    # keypoint_visible=[]
    # human_hand_keypoint=[]
    # hand_score_list=[]
    # ratio_keypoint = []
    # ratio_hand_keypoint = []
    # body_distance= []
    # hand_distance = []
    if target == 'body':
        pose = []
        for key in json_raw_info:
            values = json_raw_info[key]
            pose.append(values['pose_ratio'])
        all_info = {}
        all_info['body_pose'] = torch.Tensor(pose)
    if target == 'hand':
        pose = []
        for key in json_raw_info:
            values = json_raw_info[key]
            pose.append(values['pose_ratio'])
        all_info = {}
        all_info['hand_pose'] = torch.Tensor(pose)
    #     # if not values:######## skip bad images##########
    #     #     continue
    #     # if 'body_dis' not in values.keys(): ######## skip some img missing distance result####
    #     #     continue
    #     body_ratio.append(values['body_ratio'][0][0])######3only keep first on for convenience for now####
    #     body_distance.append(values['body_dis'])
    #     ratio_keypoint.append(values['ratio_keypoint'][0])
    #     if  'hand_ratio' in values.keys() and values['hand_ratio']>0:
    #         hand_ratio.append(values['hand_ratio'])
    #         hand_distance.append(values['hand_dis'])
    #         ratio_hand_keypoint.append(values['ratio_hand_keypoint'][0])
    #     human_body_keypoint.append(values['body_pose'])
    #     keypoint_score.append(values['score'][0])
    #     keypoint_visible.append(values['visible'][0])
    #     if 'hand_pose' in values.keys() and not isinstance(values['hand_pose'],int):
    #         #human_hand_keypoint.append(values['hand_pose'])
    #         human_hand_keypoint +=values['hand_pose']########## in case one21 hand or 42 two hand#####
    #     if 'hand_score' in values.keys() and not isinstance(values['hand_score'],int):
    #         hand_score_list.append(values['hand_score'][0])
    # body_ratio = torch.tensor(body_ratio)
    # hand_ratio = torch.tensor(hand_ratio)
    # body_distance = torch.tensor(body_distance)
    # hand_distance = torch.tensor(hand_distance)
    # human_body_keypoint = torch.tensor(human_body_keypoint)
    # keypoint_score = torch.tensor(keypoint_score)
    # keypoint_visible = torch.tensor(keypoint_visible)
    # human_hand_keypoint = torch.tensor(human_hand_keypoint)
    # hand_score_list = torch.tensor(hand_score_list)
    # ratio_keypoint = torch.tensor(ratio_keypoint)
    # ratio_hand_keypoint  = torch.tensor(ratio_hand_keypoint)
    # all_info = {}
    # all_info['body_ratio'] = body_ratio
    # all_info['hand_ratio'] = hand_ratio
    # all_info['human_body_keypoint'] = human_body_keypoint
    # all_info['keypoint_score'] = keypoint_score
    # all_info['keypoint_visible'] = keypoint_visible
    # all_info['human_hand_keypoint'] = human_hand_keypoint
    # all_info['hand_score_list'] = hand_score_list
    # all_info['ratio_keypoint'] = ratio_keypoint
    # all_info['ratio_hand_keypoint'] = ratio_hand_keypoint
    # all_info['hand_distance'] = hand_distance
    # all_info['body_distance'] = body_distance
    return all_info
def compute_dis_real_syn(syn_type,hoi_type,real_num,syn_num,data_path):
    ###################seperately generate real data info and generate info per hoi category#################
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    real_name = f'hicodet_{real_num}'
    #######these are collections for all hoi classes in corresponding split#######3
    all_hoi_collection = {}
    body_dis = []
    hand_dis = []
    ratio_body_dis =[]
    ratio_hand_dis = []
    real_body_ratio = []
    real_hand_ratio = []
    syn_body_ratio = []
    syn_hand_ratio = []
    real_body_pixel_distance = []
    real_hand_pixel_distance = []
    syn_body_pixel_distance = []
    syn_hand_pixel_distance = []
    syn_root = osp.join(data_path,f'{hoi_type}/{syn_type}')
    for hoi_idx in hoi_idxs.keys():
        #print(f'Collecting information for hoi class {hoi_idx}: {hoi_idxs[hoi_idx]}.\n')
        real_json_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{real_name}/pose_result_{hoi_idx}.json'
        #real_json_pth = osp.join(data_path,f'{hoi_type}/{real_name}/pose_result_{hoi_idx}.json')
        with open(real_json_pth,'r') as f:
            real_info = json.load(f)
        #syn_json_pth =  f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_name}/pose_result_{hoi_idx}.json'
        #syn_json_pth = osp.join(data_path,f'{hoi_type}/{syn_name}/pose_result_{hoi_idx}.json')
        syn_json_path = osp.join(syn_root,f'{hoi_idx}/pose.json')
        with open(syn_json_path,'r') as f:
            syn_info = json.load(f)
        syn_hand_json_path = osp.join(syn_root,f'{hoi_idx}/hand_pose.json')
        with open(syn_hand_json_path,'r') as f:
            syn_hand_info = json.load(f)
        real_pose_info = collect_real_json_info_for_single_hoi(real_info)
        syn_pose_info = collect_json_info_for_single_hoi(syn_info,target='body')
        syn_hand_pose_info = collect_json_info_for_single_hoi(syn_hand_info,target='hand')
        hoi_idx_collection = {}
        # try:
        body_pose_dis = wasserstein_distance(real_pose_info['human_body_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['body_pose'].reshape(-1,2).cpu().numpy().flatten())
        hand_pose_dis = wasserstein_distance(real_pose_info['human_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_hand_pose_info['hand_pose'].reshape(-1,2).cpu().numpy().flatten())
        # except:
        #     continue########skip bad hoi samples
        #try:
        ratio_body_pose_dis = wasserstein_distance(real_pose_info['ratio_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['body_pose'].reshape(-1,2).cpu().numpy().flatten())
        ratio_hand_pose_dis = wasserstein_distance(real_pose_info['ratio_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_hand_pose_info['hand_pose'].reshape(-1,2).cpu().numpy().flatten())
        hoi_idx_collection['ratio_keypoint_dis']  =ratio_body_pose_dis
        hoi_idx_collection['ratio_hand_keypoint_dis'] = ratio_hand_pose_dis
        hoi_idx_collection['keypoint_dis']  =body_pose_dis
        hoi_idx_collection['hand_keypoint_dis'] = hand_pose_dis
            #########TODO: add 1.human+ object pose distribution distance and 2.keypoint confidence score 3.animal pose score####################
        # except:
        #     hoi_idx_collection['ratio_keypoint_dis']  =torch.zeros((0,2))
        #     hoi_idx_collection['ratio_hand_keypoint_dis'] = torch.zeros((0,2))
        # real_body_ratio_mean = torch.mean(real_pose_info['body_ratio'])
        # real_hand_ratio_mean =  torch.mean(real_pose_info['hand_ratio'])
        # syn_body_ratio_mean = torch.mean(syn_pose_info['body_ratio'])
        # syn_hand_ratio_mean = torch.mean(syn_pose_info['hand_ratio'])
        
        # real_body_distance_mean = torch.mean(real_pose_info['body_distance'])
        # real_hand_distance_mean = torch.mean(real_pose_info['hand_distance'])
        # syn_body_distance_mean = torch.mean(syn_pose_info['body_distance'])
        # syn_hand_distance_mean = torch.mean(syn_pose_info['hand_distance'])
        # hoi_idx_collection['body_pose_dis'] = body_pose_dis
        # hoi_idx_collection['hand_pose_dis'] = hand_pose_dis

        # hoi_idx_collection['real_body_ratio_mean'] = real_body_ratio_mean.item()
        # hoi_idx_collection['real_hand_ratio_mean'] = real_hand_ratio_mean.item()
        # hoi_idx_collection['syn_body_ratio_mean'] = syn_body_ratio_mean.item()
        # hoi_idx_collection['syn_hand_ratio_mean'] = syn_hand_ratio_mean.item()
        # hoi_idx_collection['real_body_distance_mean'] = real_body_distance_mean.item()
        # hoi_idx_collection['real_hand_distance_mean'] = real_hand_distance_mean.item()
        # hoi_idx_collection['syn_body_distance_mean'] = syn_body_distance_mean.item()
        # hoi_idx_collection['syn_hand_distance_mean'] = syn_hand_distance_mean.item()
        body_dis.append(body_pose_dis)
        hand_dis.append(hand_pose_dis)
        
        ratio_body_dis.append(ratio_body_pose_dis)
        ratio_hand_dis.append(ratio_hand_pose_dis)
        #real_body_ratio.append(real_body_ratio_mean.item())
        # real_hand_ratio.append(real_hand_ratio_mean.item())
        # syn_body_ratio.append(syn_body_ratio_mean.item())
        # syn_hand_ratio.append(syn_hand_ratio_mean.item())
        # real_body_pixel_distance.append(real_body_distance_mean.item())
        # real_hand_pixel_distance.append(real_hand_distance_mean.item())
        # syn_body_pixel_distance.append(syn_body_distance_mean.item())
        # syn_hand_pixel_distance.append(syn_hand_distance_mean.item())
        # with open(osp.join(data_path,f'{hoi_type}/{syn_name}/compare_{hoi_idx}.json'),'w') as file:
        # #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_name}/compare_{hoi_idx}.json','w') as file:
        #     json.dump(hoi_idx_collection,file)
        all_hoi_collection[hoi_idx] = hoi_idx_collection
        #print(f'Finish collecting information for hoi class {hoi_idx}.\n')
    # with open(f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_name}/compare_all.json','w') as f:
    #with open(osp.join(data_path,f'{hoi_type}/{syn_name}/compare_all.json'),'w') as f:
    #    json.dump(all_hoi_collection,f)
    #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_name}/all_result.txt','w') as f:
    print(f'ratio_body_dis is {ratio_body_dis}, and average is {sum(ratio_body_dis)/len(ratio_body_dis)}.\n')
    print(f'ratio_hand_dis is {ratio_hand_dis}, and average is {sum(ratio_hand_dis)/len(ratio_hand_dis)}.\n')
    print(f'body_dis is {ratio_body_dis}, and average is {sum(body_dis)/len(body_dis)}.\n')
    print(f'hand_dis is {ratio_hand_dis}, and average is {sum(hand_dis)/len(hand_dis)}.\n')
    # with open(osp.join(data_path,f'{hoi_type}/{syn_name}/all_result.txt'),'w') as f:
    #     f.write(f'Final average for this {hoi_type} is as follw\n')
    #     f.write(f'body_dis is {body_dis}, and average is {sum(body_dis)/len(body_dis)}.\n')
    #     f.write(f'hand_dis is {hand_dis}, and average is {sum(hand_dis)/len(hand_dis)}.\n')
    #     f.write(f'ratio_body_dis is {ratio_body_dis}, and average is {sum(ratio_body_dis)/len(ratio_body_dis)}.\n')
    #     f.write(f'ratio_hand_dis is {ratio_hand_dis}, and average is {sum(ratio_hand_dis)/len(ratio_hand_dis)}.\n')
    #     f.write(f'real_body_ratio is {real_body_ratio}, and average is {sum(real_body_ratio)/len(real_body_ratio)}.\n')
    #     f.write(f'real_hand_ratio is {real_hand_ratio}, and average is {sum(real_hand_ratio)/len(real_hand_ratio)}.\n')
    #     f.write(f'syn_body_ratio is {syn_body_ratio}, and average is {sum(syn_body_ratio)/len(syn_body_ratio)}.\n')
    #     f.write(f'syn_hand_ratio is {syn_hand_ratio}, and average is {sum(syn_hand_ratio)/len(syn_hand_ratio)}.\n')
    #     f.write(f'real_body_distance is {real_body_pixel_distance}, and average is {sum(real_body_pixel_distance)/len(real_body_pixel_distance)}.\n')
    #     f.write(f'real_hand_distance is {real_hand_pixel_distance}, and average is {sum(real_hand_pixel_distance)/len(real_hand_pixel_distance)}.\n')
    #     f.write(f'syn_body_distance is {syn_body_pixel_distance}, and average is {sum(syn_body_pixel_distance)/len(real_body_pixel_distance)}.\n')
    #     f.write(f'syn_hand_distance is {syn_hand_pixel_distance}, and average is {sum(syn_hand_pixel_distance)/len(real_hand_pixel_distance)}.\n')
    #     f.close()


def compute_animal_dis(syn_type,syn_num,real_num,data_path):
    ###################seperately generate real data info and generate info per hoi category#################
    hoi_idxs = animal_idx
    #######these are collections for all hoi classes in corresponding split#######3
    all_hoi_collection = {}
    all_animal_ratio_dis = []
    real_body_ratio = []
    syn_body_ratio = []
    real_name = f'hicodet_{real_num}'
    syn_name = syn_type + '_' + str(syn_num)
    if real_num==20:
        real_name = 'hicodet'
    if syn_num==20:
        syn_name = syn_type
    for hoi_idx in hoi_idxs.keys():
        print(f'Collecting information for hoi class {hoi_idx}: {hoi_idxs[hoi_idx]}.\n')
        #real_animal_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/animal_{hoi_idx}.json'
        real_animal_pth = osp.join(data_path,f'animal/{real_name}/animal_{hoi_idx}.json')
        with open(real_animal_pth,'r') as f:
            real_animal = json.load(f)
        #syn_animal_pth =  f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/animal_{hoi_idx}.json'
        syn_animal_pth = osp.join(data_path,f'animal/{syn_name}/animal_{hoi_idx}.json')
        with open(syn_animal_pth,'r') as f:
            syn_animal = json.load(f)
        # real_joint_info,real_ratio_joint_info = collect_joint_info(real_info,real_animal)
        # syn_joint_info,syn_ratio_joint_info = collect_joint_info(syn_info,syn_animal)
        real_animal_info = collect_animal_ratio_pose(real_animal)
        syn_animal_info = collect_animal_ratio_pose(syn_animal)
        try:
            animal_ratio_pose_dis = wasserstein_distance(real_animal_info.reshape(-1,2).cpu().numpy().flatten(),syn_animal_info.reshape(-1,2).cpu().numpy().flatten())
        except:
            continue########skip bad hoi images
            #########TODO: add 1.human+ object pose distribution distance and 2.keypoint confidence score 3.animal pose score####################
        hoi_idx_collection = {}
        hoi_idx_collection['animal_pose_dis'] = animal_ratio_pose_dis
        all_animal_ratio_dis.append(animal_ratio_pose_dis)
        # with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/animal_dis_{hoi_idx}.json','w') as file:
        with open(osp.join(data_path,f'animal/{syn_name}/animal_dis_{hoi_idx}.json'),'w') as file:
            json.dump(hoi_idx_collection,file)
        all_hoi_collection[hoi_idx] = hoi_idx_collection
        print(f'Finish collecting information for hoi class {hoi_idx}.\n')
    #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/animal_dis_all.json','w') as f:
    with open(osp.join(data_path,f'animal/{syn_name}/animal_dis_all.json'),'w') as file:
        json.dump(all_hoi_collection,f)
    #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/animal_result.txt','w') as f:
    with open(osp.join(data_path,f'animal/{syn_name}/animal_result.txt'),'w') as f:
        f.write(f'animal_ratio_pose_dis is {all_animal_ratio_dis}, and average is {sum(all_animal_ratio_dis)/len(all_animal_ratio_dis)}.\n')
        f.close()
        
def compute_joint_dis(syn_type,syn_num,real_num,data_pah):
    ###################seperately generate real data info and generate info per hoi category#################
    hoi_idxs = animal_idx
    #######these are collections for all hoi classes in corresponding split#######3
    all_hoi_collection = {}
    all_joint_body_dis = []
    all_joint_ratio_body_dis = []
    real_body_ratio = []
    syn_body_ratio = []
    real_name = f'hicodet_{real_num}'
    syn_name = syn_type + '_' + str(syn_num)
    if real_num==20:
        real_name = 'hicodet'
    if syn_num==20:
        syn_name = syn_type
    for hoi_idx in hoi_idxs.keys():
        print(f'Collecting information for hoi class {hoi_idx}: {hoi_idxs[hoi_idx]}.\n')
        # real_json_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/pose_result_{hoi_idx}.json'
        # real_animal_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/animal_{hoi_idx}.json'
        real_json_pth = osp.join(data_pah,f'animal/{real_name}/pose_result_{hoi_idx}.json')
        real_animal_pth = osp.join(data_pah,'animal/{real_name}/animal_{hoi_idx}.json')
        with open(real_json_pth,'r') as f:
            real_info = json.load(f)
        with open(real_animal_pth,'r') as f:
            real_animal = json.load(f)
        # syn_json_pth =  f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/pose_result_{hoi_idx}.json'
        # syn_animal_pth =  f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/animal_{hoi_idx}.json'
        syn_json_pth = osp.join(data_pah,f'animal/{syn_name}/pose_result_{hoi_idx}.json')
        syn_animal_pth = osp.join(data_pah,'animal/{syn_name}/animal_{hoi_idx}.json')
        with open(syn_json_pth,'r') as f:
            syn_info = json.load(f)
        with open(syn_animal_pth,'r') as f:
            syn_animal = json.load(f)
        real_joint_info,real_ratio_joint_info = collect_joint_info(real_info,real_animal)
        syn_joint_info,syn_ratio_joint_info = collect_joint_info(syn_info,syn_animal)

        try:
            joint_pose_dis = wasserstein_distance(real_joint_info.reshape(-1,2).cpu().numpy().flatten(),syn_joint_info.reshape(-1,2).cpu().numpy().flatten())
            joint_ratio_pose_dis = wasserstein_distance(real_ratio_joint_info.reshape(-1,2).cpu().numpy().flatten(),syn_ratio_joint_info.reshape(-1,2).cpu().numpy().flatten())
        except:
            continue########skip bad hoi images
            #########TODO: add 1.human+ object pose distribution distance and 2.keypoint confidence score 3.animal pose score####################
        hoi_idx_collection = {}
        hoi_idx_collection['joint_pose_dis'] = joint_pose_dis
        hoi_idx_collection['joint_ratio_pose_dis'] = joint_ratio_pose_dis
        all_joint_body_dis.append(joint_pose_dis)
        all_joint_ratio_body_dis.append(joint_ratio_pose_dis)
        #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/joint_compare_{hoi_idx}.json','w') as file:
        with open(osp.join(data_pah,f'animal/{syn_name}/joint_compare_{hoi_idx}.json'),'w') as file:
            json.dump(hoi_idx_collection,file)
        all_hoi_collection[hoi_idx] = hoi_idx_collection
        print(f'Finish collecting information for hoi class {hoi_idx}.\n')
    #with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/joint_compare_all.json','w') as f:
    with open(osp.join(data_pah,f'animal/{syn_name}/joint_compare_all.json'),'w') as f:
        json.dump(all_hoi_collection,f)
    # with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_name}/joint_result.txt','w') as f:
    with open(osp.join(data_pah,'animal/{syn_name}/joint_result.txt'),'w') as f:
        f.write(f'joint_body_dis is {all_joint_body_dis}, and average is {sum(all_joint_body_dis)/len(all_joint_body_dis)}.\n')
        f.write(f'joint_ratio_body_dis is {all_joint_ratio_body_dis}, and average is {sum(all_joint_ratio_body_dis)/len(all_joint_ratio_body_dis)}.\n')
        f.close()


# def collect_keypoint_score(keypoint_info):
#     #hand_scores =[]
#     body_score = []
#     for key in keypoint_info:
#         values = keypoint_info[key]
#         if not values:
#             continue
#         ############seperate count on body and hand##########
#         if 'hand_pose' in values.keys() and 'hand_score' in values.keys() and not isinstance(values['hand_pose'],int) and not isinstance(values['hand_score'],int):
#             #hand_scores.append(values['hand_score'][0])
#             hand_scores+= values['hand_score'][0]
#         visible_info = values['visible'][0]
#         # keep_idx = visible_info>0.5
#         # keep_body_score = values['score'][0][keep_idx]
#         indexes = [i for i, x in enumerate(visible_info) if x > 0.5]
#         keep_body_score = [values['score'][0][i] for i in indexes]
#         #body_score.append(keep_body_score)
#         body_score += keep_body_score
#     return body_score, hand_scores

def collect_animal_keypoint_score(keypoint_info):
    body_score = []
    for key in keypoint_info:
        values = keypoint_info[key]
        if not values:
            continue
        visible_info = values['visible'][0]
        # keep_idx = visible_info>0.5
        # keep_body_score = values['score'][0][keep_idx]
        indexes = [i for i, x in enumerate(visible_info) if x > 0.5]
        keep_body_score = [values['score'][0][i] for i in indexes]
        #body_score.append(keep_body_score)
        body_score += keep_body_score
    return body_score
def get_img_hw(img_pth):
    with Image.open(img_pth) as img:
        h, w = img.size
    return h,w

def get_keypoint_ratio(h,w,keypoint):
    ratio = keypoint
    if ratio.dim()==3:
        ratio[0,:, 0] /= h
        ratio[0,:,1] /=w
    elif ratio.dim()==2:
        ratio[:, 0] /= h
        ratio[:,1] /=w
    return ratio

def get_keypoint_ratio_using_bbox(keypoint,bbox):
    if bbox.dim()==3:#########n*1*4,for hand#####
        hand_num = bbox.shape[0]
        hand_result = []
        keypoints_array = np.array(keypoint)
        for i in range(hand_num):
            x_min, y_min, x_max, y_max = bbox[i,0,:]
            x_min = x_min.item()
            y_min = y_min.item()
            x_max = x_max.item()
            y_max = y_max.item()
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            normalized_keypoints = (keypoints_array[i] - np.array([x_min, y_min])) / np.array([bbox_width, bbox_height])
            #normalized_keypoints = normalized_keypoints[np.newaxis, :]
            transfer_normalized_keypoints = normalized_keypoints - normalized_keypoints[0,:]
            hand_result.append(torch.tensor(transfer_normalized_keypoints))
        return torch.stack(hand_result,dim=0)
    else:  #########1*4, for body######
        x_min, y_min, x_max, y_max = bbox[0]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        if keypoint.shape[0] >1:
            keypoint = keypoint[0].unsqueeze(0)  ####### we only keep the first human#####
        if keypoint.dim()==3: #######for body pose#########
            keypoints_array = np.array(keypoint).squeeze(axis=0)
            normalized_keypoints = (keypoints_array - np.array([x_min, y_min])) / np.array([bbox_width, bbox_height])
            normalized_keypoints = normalized_keypoints[np.newaxis, :]
            transfer_normalized_keypoints = normalized_keypoints - normalized_keypoints[0,0,:]
        elif keypoint.dim()==2:############## for hand pose##########
            keypoints_array = np.array(keypoint)
            normalized_keypoints = (keypoints_array - np.array([x_min, y_min])) / np.array([bbox_width, bbox_height])
            transfer_normalized_keypoints = normalized_keypoints - normalized_keypoints[0,:] 
        return torch.tensor(transfer_normalized_keypoints)

def collect_keypoints(hoi_type,data_path,syn_model,target):
    data_root = osp.join(data_path,f'{hoi_type}/{syn_model}')
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    pose_all = {}
    if target == 'body':
        for hoi_idx in hoi_idxs.keys():
            hoi_dir = osp.join(data_root,f'{hoi_idx}')
            files = [f for f in os.listdir(hoi_dir) if os.path.isfile(os.path.join(hoi_dir, f))]
            file_seed_names = [os.path.basename(file) for file in files]
            pose_all_info_idx = {} 
            for name in file_seed_names:
                img_pth = osp.join(hoi_dir,name)
                if not img_pth.endswith('.png'):
                    continue
                info ={}
                result_generator = inferencer(img_pth,shows = False,return_datasamples=True)
                result = [result for result in result_generator][0]['predictions']
                keypoint_list = torch.tensor(result[0].pred_instances.keypoints)
                keypoint_score = torch.tensor(result[0].pred_instances.keypoint_scores)
                keypoint_visible = torch.tensor(result[0].pred_instances.keypoints_visible)
                human_box = torch.tensor(result[0].pred_instances.bboxes)
                human_box_score = torch.tensor(result[0].pred_instances.bbox_scores)
                h,w = get_img_hw(img_pth)
                ratio_keypoint = get_keypoint_ratio_using_bbox(keypoint_list,human_box)  
                info['pose_ratio']  = ratio_keypoint.tolist()
                info['keypoint_score'] = keypoint_score.tolist()
                info['keypoint_visible'] = keypoint_visible.tolist()
                pose_all_info_idx[name] = info
            with open(osp.join(hoi_dir,'pose.json'),'w') as f:
                json.dump(pose_all_info_idx,f)
            print(f'finish pose info statistic for {hoi_idx}.')
    if target == 'hand':
        for hoi_idx in hoi_idxs.keys():
            hoi_dir = osp.join(data_root,f'{hoi_idx}')
            files = [f for f in os.listdir(hoi_dir) if os.path.isfile(os.path.join(hoi_dir, f))]
            file_seed_names = [os.path.basename(file) for file in files]
            pose_all_info_idx = {} 
            for name in file_seed_names:
                if not name.endswith('png'):
                    continue
                img_pth = osp.join(hoi_dir,name)
                info ={}
                result_generator = hand_inferencer(img_pth,shows = False,return_datasamples=True)
                result = [result for result in result_generator][0]['predictions']
                keypoint_list = torch.tensor(result[0].pred_instances.keypoints)
                keypoint_score = torch.tensor(result[0].pred_instances.keypoint_scores)
                keypoint_visible = torch.tensor(result[0].pred_instances.keypoints_visible)
                human_box = torch.tensor(result[0].pred_instances.bboxes)
                human_box_score = torch.tensor(result[0].pred_instances.bbox_scores)
                h,w = get_img_hw(img_pth)
                ratio_keypoint = get_keypoint_ratio_using_bbox(keypoint_list,human_box)  
                info['pose_ratio']  = ratio_keypoint.tolist()
                info['keypoint_score'] = keypoint_score.tolist()
                info['keypoint_visible'] = keypoint_visible.tolist()
                pose_all_info_idx[name] = info
            with open(osp.join(hoi_dir,'hand_pose.json'),'w') as f:
                json.dump(pose_all_info_idx,f)
            print(f'finish pose info statistic for {hoi_idx}.')
def collect_keypoint_score(keypoint_info):
    body_score = []
    for key in keypoint_info:
        values = keypoint_info[key]
        if not values:
            continue
        visible_info = values['keypoint_visible'][0]
        # keep_idx = visible_info>0.5
        # keep_body_score = values['score'][0][keep_idx]
        indexes = [i for i, x in enumerate(visible_info) if x > 0.3]
        keep_body_score = [values['keypoint_score'][0][i] for i in indexes]
        #body_score.append(keep_body_score)
        body_score += keep_body_score
    return body_score

def collect_keypoint_hand_score(keypoint_info):
    body_score = []
    for key in keypoint_info:
        values = keypoint_info[key]
        if not values:
            continue
        visible_info = values['keypoint_visible'][0]
        # keep_idx = visible_info>0.5
        # keep_body_score = values['score'][0][keep_idx]
        indexes = [i for i, x in enumerate(visible_info) if x > 0]
        keep_body_score = [values['keypoint_score'][0][i] for i in indexes]
        #body_score.append(keep_body_score)
        body_score += keep_body_score
    return body_score

def keypoint_score_compare(hoi_type,real_num,syn_num,data_path,syn_model,target):
    #########compare keypoint score for three model,sd,type2, and final#########
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    sd_img_root_pth =osp.join(data_path,f'{hoi_type}/sd')
    pia_img_root_pth = osp.join(data_path,f'{hoi_type}/pia')
    sd_body_all =[]
    sd_hand_all =[]
    hand_body_all =[]
    final_body_all = []
    final_hand_all = []
    hand_hand_all = []
    if target =='body':
        for hoi_idx in hoi_idxs.keys():
            sd_config = os.path.join(sd_img_root_pth,f'{hoi_idx}/pose.json')
            with open(sd_config,'r') as f:
                sd_pose = json.load(f)
            final_config = os.path.join(pia_img_root_pth,f'{hoi_idx}/pose.json')
            with open(final_config,'r') as f:
                final_pose = json.load(f)
            # hand_config = os.path.join(hand_img_root_pth,f'pose_result_{hoi_idx}.json')
            # with open(hand_config,'r') as f:
            #     hand_pose = json.load(f)
            sd_scores = collect_keypoint_score(sd_pose)
            final_scores = collect_keypoint_score(final_pose)
            # hand_scores,hand_hand = collect_keypoint_score(hand_pose)
            sd_body_all.append(sum(sd_scores)/len(sd_scores))
            final_body_all.append(sum(final_scores)/len(final_scores))
        print(f'sd pcs score is {sum(sd_body_all)/len(sd_body_all)}')   
        print(f'pia pcs score is {sum(final_body_all)/len(final_body_all)}') 
    if target == 'hand':
        for hoi_idx in hoi_idxs.keys():
            sd_config = os.path.join(sd_img_root_pth,f'{hoi_idx}/hand_pose.json')
            with open(sd_config,'r') as f:
                sd_pose = json.load(f)
            final_config = os.path.join(pia_img_root_pth,f'{hoi_idx}/hand_pose.json')
            with open(final_config,'r') as f:
                final_pose = json.load(f)
            # hand_config = os.path.join(hand_img_root_pth,f'pose_result_{hoi_idx}.json')
            # with open(hand_config,'r') as f:
            #     hand_pose = json.load(f)
            sd_scores = collect_keypoint_hand_score(sd_pose)
            final_scores = collect_keypoint_hand_score(final_pose)
            # hand_scores,hand_hand = collect_keypoint_score(hand_pose)
            sd_body_all.append(sum(sd_scores)/len(sd_scores))
            final_body_all.append(sum(final_scores)/len(final_scores))
        print(f'sd pcs score for hand is {sum(sd_body_all)/len(sd_body_all)}')   
        print(f'pia pcs score for hand is {sum(final_body_all)/len(final_body_all)}') 

parser = argparse.ArgumentParser()
parser.add_argument('--syn_type',type=str,default='sd')
parser.add_argument('--hoi_type',type=str,default='connect')
parser.add_argument('--compare_keypoint',default=False,action='store_true')
parser.add_argument('--compare_animal_keypoint',default=False,action='store_true')
parser.add_argument('--pose_dis',default=False,action='store_true')
parser.add_argument('--animal_keypoint_dis',default=False,action='store_true')
parser.add_argument('--joint_keypoint_dis',default=False,action='store_true')
parser.add_argument('--real_num',type = int,default=200)
parser.add_argument('--syn_num',type = int,default=20)
parser.add_argument('--data_path',type=str,default='/scratch/yangdejie/xz/ReVersion/clean_hoi')
parser.add_argument('--compare_hand',default=False,action = 'store_true')
args = parser.parse_args()
#compute_dis_real_syn(syn_type =args.syn_type , hoi_type = args.hoi_type,real_num = args.real_num, syn_num = args.syn_num)
if args.compare_keypoint:
    collect_keypoints(hoi_type = args.hoi_type,data_path = args.data_path,syn_model = 'sd',target='body')
    collect_keypoints(hoi_type = args.hoi_type,data_path = args.data_path,syn_model = 'pia',target='body')
    keypoint_score_compare(args.hoi_type,real_num = args.real_num, syn_num = args.syn_num,data_path = args.data_path,syn_model = args.syn_type,target='body')
if args.compare_hand:
    collect_keypoints(hoi_type = args.hoi_type,data_path = args.data_path,syn_model = 'sd',target='hand')
    collect_keypoints(hoi_type = args.hoi_type,data_path = args.data_path,syn_model = 'pia',target='hand')
    keypoint_score_compare(args.hoi_type,real_num = args.real_num, syn_num = args.syn_num,data_path = args.data_path,syn_model = args.syn_type,target='hand')
if args.pose_dis:
    compute_dis_real_syn(syn_type=args.syn_type,hoi_type=args.hoi_type,real_num=args.real_num,syn_num=args.syn_num,data_path=args.data_path)
if args.joint_keypoint_dis:
    compute_joint_dis(syn_type = args.syn_type,syn_num = args.syn_num,real_num = args.real_num,data_path = args.data_path)
if args.animal_keypoint_dis:
    compute_animal_dis(syn_type = args.syn_type,syn_num = args.syn_num,real_num = args.real_num,data_path = args.data_path)