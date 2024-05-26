from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
from hoi_configs.human import human_idx
import json
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
image_root_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/images/train2015/'
config_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/annotations/trainval_hico.json'
from pose_api import call_pose_api, call_hand_pose_api
import cv2
import numpy as np
from scipy.spatial import distance
from scipy import ndimage
import argparse
import PIL
from PIL import Image
import numpy as np
import os.path as osp
from scipy.stats import wasserstein_distance
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
inferencer = MMPoseInferencer(pose2d='human')
hand_inferencer = MMPoseInferencer(pose2d='hand')
animal_inferencer = MMPoseInferencer(pose2d='animal')
# from hand_pose import generate_cannny_obj, connect_judgement
def get_obj_mask(img_pth):
    init_image = Image.open(img_pth)
    init_image_np = np.array(init_image)
    try:
        outputs = predictor(init_image_np)
        v = Visualizer(init_image_np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        is_person = (outputs["instances"].pred_classes == 0).cpu().numpy()
        is_person = np.where(is_person)[0]
        not_person = (outputs["instances"].pred_classes != 0).cpu().numpy()
        not_person = np.where(not_person)[0]

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        import PIL
        if len(is_person) == 0:
            print("MaskRCNN: No person is detected in the image.")
            mask = np.zeros_like(init_image_np)[:, :, 0]
            mask_image = PIL.Image.fromarray(mask)
            
        else:
            mask = np.zeros_like(outputs["instances"].pred_masks[0].cpu().numpy())
            for idx in is_person:
                mask += outputs["instances"].pred_masks[idx].cpu().numpy()
            mask = mask > 0
            mask_image = PIL.Image.fromarray(mask)
            #mask_image.save('./images/person_mask.png')
        non_person_mask = np.zeros_like(mask)
        if len(not_person) > 0:
            for idx in not_person:
                non_person_mask += outputs["instances"].pred_masks[idx].cpu().numpy()
            non_person_mask = non_person_mask > 0
        non_person_mask_image = PIL.Image.fromarray(non_person_mask)
    except:
        non_person_mask = None
    return non_person_mask
from scipy.spatial.distance import cdist
def generate_cannny_obj(origin_mask): ####origin mask:512*512,bool type
    mask = origin_mask.astype(np.uint8)
    if mask.max()>0:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        coordinates = contours[0]
        coordinates = coordinates.reshape(-1, 2)
    else:
        coordinates = torch.zeros((1,2))
    return coordinates

def connect_judgement(human_keypoint,obj_canny,threshold=0):
    distance_matrix = cdist(human_keypoint, obj_canny, metric='euclidean')
    min_distance = np.min(distance_matrix)
    min_distance_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    return min_distance
    # if min_distance < threshold:
    #     return 1
    # return 0
def compute_bbox_score(bboxes):
    x1, y1, x2, y2 = torch.chunk(bboxes, chunks=4, dim=1)
    areas = (x2 - x1 ) * (y2 - y1 )
    return areas

def cal_animal_pose(img_pth):
    result_generator = animal_inferencer(img_pth,shows = False,return_datasamples=True)
    result = [result for result in result_generator][0]['predictions']
    keypoint_list = torch.tensor(result[0].pred_instances.keypoints)
    keypoint_score = torch.tensor(result[0].pred_instances.keypoint_scores)
    keypoint_visible = torch.tensor(result[0].pred_instances.keypoints_visible)
    human_box = torch.tensor(result[0].pred_instances.bboxes)
    human_box_score = torch.tensor(result[0].pred_instances.bbox_scores)
    animal_pose_keypoint = keypoint_list[0]
    return animal_pose_keypoint,keypoint_score,keypoint_visible,human_box
def cal_dis_between_h_o(img_pth):
    ######generate object canny
    obj_mask = get_obj_mask(img_pth)
    if_obj_exist = True
    if obj_mask is None:
       if_obj_exist =False 
    if if_obj_exist:
        obj_corrd = generate_cannny_obj(obj_mask)
        if obj_corrd.max() == 0 :
            if_obj_exist = False
    else:
        obj_corrd = None
    result_generator = inferencer(img_pth,shows = False,return_datasamples=True)
    result = [result for result in result_generator][0]['predictions']
    keypoint_list = torch.tensor(result[0].pred_instances.keypoints)
    keypoint_score = torch.tensor(result[0].pred_instances.keypoint_scores)
    keypoint_visible = torch.tensor(result[0].pred_instances.keypoints_visible)
    human_box = torch.tensor(result[0].pred_instances.bboxes)
    human_box_score = torch.tensor(result[0].pred_instances.bbox_scores)
    h,w = get_img_hw(img_pth)
    ratio_keypoint = get_keypoint_ratio_using_bbox(keypoint_list,human_box)   ########## change to bbox normalized keypoint######
    hand_result_generator = hand_inferencer(img_pth,shows = False)
    hand_result = [result for result in hand_result_generator][0]['predictions']

    # hand_keypoint_list = torch.tensor(hand_result[0][0]['keypoints'])
    if hand_result is not None:
        hand_keypoint_list = [item['keypoints'] for item in hand_result[0]]
        hand_score_list = [item['keypoint_scores'] for item in hand_result[0]]
        hand_bbox_list = [item['bbox'] for item in hand_result[0]]
        hand_bbox_score_list = [item['bbox_score'] for item in hand_result[0]]
        hand_keypoint_list = torch.tensor(hand_keypoint_list)
        hand_score_list = torch.tensor(hand_score_list)
        # hand_bbox_list = torch.tensor(hand_bbox_list).reshape(-1,4)
        hand_bbox_list = torch.tensor(hand_bbox_list)
        hand_bbox_score_list = torch.tensor(hand_bbox_score_list)
        human_hand_keypoint = hand_keypoint_list.reshape(-1,2)
        #ratio_hand_keypoint = get_keypoint_ratio(h,w,human_hand_keypoint)
        ratio_hand_keypoint = get_keypoint_ratio_using_bbox(hand_keypoint_list,hand_bbox_list)
        if if_obj_exist:
            hand_dis = connect_judgement(human_hand_keypoint,obj_corrd)
            hand_bbox_area = compute_bbox_score(hand_bbox_list.reshape(-1,4))
            hand_area = torch.mean(hand_bbox_area).item()
            hand_ratio = hand_dis/hand_area
        else:
            hand_dis = 0
            hand_ratio = 0

    human_body_keypoint = keypoint_list[0]
    if if_obj_exist:
        body_dis = connect_judgement(human_body_keypoint,obj_corrd)
        body_bbox_area = compute_bbox_score(human_box)
        body_ratio = body_dis/body_bbox_area
    else:
        body_dis = 0
        body_ratio = 0
    if hand_result is not None:
        return body_ratio, hand_ratio, human_body_keypoint, keypoint_score,keypoint_visible,human_hand_keypoint,\
            hand_score_list, ratio_keypoint,ratio_hand_keypoint, body_dis,hand_dis
    return body_ratio, 0,human_body_keypoint, keypoint_score,keypoint_visible,0,0,0,0,body_dis,0
    #############################################
    
def pose_dis_cal_for_imgs(type ='animal',if_resize=False,num=20):  
    
    model_name = 'hicodet_200'        

    img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/{type}/{model_name}/' 
    type_name = type + '_200'
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi_configs/{type_name}.json','r') as f:
        img_name = json.load(f)
    pose_all = {}
    if type == 'animal':
        idx_list = animal_idx
    elif type == 'connect':
        idx_list = connect_idx
    else:
        idx_list = human_idx
    for idx in idx_list.keys():

        img_dir_pth = os.path.join(img_root_pth,str(idx))
        idx_names = img_name[str(idx)]
        pose_all_info_idx = {} 
        for name in idx_names:
            img_pth = os.path.join(img_dir_pth,name)
            body_ratio, hand_ratio, body_pose,score,visible, hand_pose, hand_score,\
                ratio_keypoint,ratio_hand_keypoint, body_dis, hand_dis= cal_dis_between_h_o(img_pth)
            key = name
            info ={}
            if not isinstance(body_ratio, int):
                info['body_ratio'] = body_ratio.tolist()
                info['body_pose'] = body_pose.tolist()
                info['score'] = score.tolist()
                info['visible'] = visible.tolist()
                info['ratio_keypoint'] = ratio_keypoint.tolist()
                info['body_dis'] = body_dis.tolist()
                if hand_ratio >0:
                    info['hand_ratio'] = hand_ratio.tolist()
                    info['hand_pose'] = hand_pose.tolist()
                    info['hand_score'] = hand_score.tolist()
                    info['ratio_hand_keypoint'] = ratio_hand_keypoint.tolist()
                    info['hand_dis'] = hand_dis.tolist()
            pose_all_info_idx[key] = info

        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/{type}/{model_name}/pose_result_{idx}.json','w') as file:
            json.dump(pose_all_info_idx,file)
        pose_all[idx] = pose_all_info_idx 

    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/{type}/{model_name}/pose_result_all.json','w') as f:
        json.dump(pose_all,f)   
        


def pose_dis_cal_for_generated_imgs(type ='animal',model='sd',num = 20,data_path='/scratch/yangdejie/xz/ReVersion/clean_hoi'): 

    img_root_pth = osp.join(data_path,f'{type}/{model}')
    pose_all = {}
    if type == 'animal':
        idx_list = animal_idx
    elif type=='connect':
        idx_list = connect_idx
    else:
        idx_list = human_idx
    for idx in idx_list.keys():

        img_dir_pth = os.path.join(img_root_pth,str(idx))
        ########## step over all seed images for this hoi idx###########
        files = [f for f in os.listdir(img_dir_pth) if os.path.isfile(os.path.join(img_dir_pth, f))]
        file_seed_names = [os.path.basename(file) for file in files]
        pose_all_info_idx = {} 
        for name in file_seed_names:
            img_pth = os.path.join(img_dir_pth,name)
            if not img_pth.endswith('.png'):
                continue
            if os.path.getsize(img_pth) < 1024*10:######### we skip over failure cases########
                continue
            body_ratio, hand_ratio, body_pose,score,visible, hand_pose, hand_score,\
                ratio_keypoint,ratio_hand_keypoint, body_dis, hand_dis= cal_dis_between_h_o(img_pth)
            key = name
            info ={}
            if not isinstance(body_ratio, int):
                info['body_ratio'] = body_ratio.tolist()
                info['body_pose'] = body_pose.tolist()
                info['score'] = score.tolist()
                info['visible'] = visible.tolist()
                info['ratio_keypoint'] = ratio_keypoint.tolist()
                info['body_dis']= body_dis.tolist()
                if hand_ratio >0:
                    info['hand_ratio'] = hand_ratio.tolist()
                    info['hand_pose'] = hand_pose.tolist()
                    info['hand_score'] = hand_score.tolist()
                    info['ratio_hand_keypoint'] = ratio_hand_keypoint.tolist() 
                    info['hand_dis'] = hand_dis.tolist()
            pose_all_info_idx[key] = info
        model_name = model
        with open(osp.join(img_dir_pth,f'hodd.json'),'w') as file:
             json.dump(pose_all_info_idx,file)
        pose_all[idx] = pose_all_info_idx 

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

def get_keypoint_ratio_using_bbox_animal(keypoint,bbox):
    
    x_min, y_min, x_max, y_max = bbox[0]
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    keypoints_array = np.array(keypoint)
    normalized_keypoints = (keypoints_array - np.array([x_min, y_min])) / np.array([bbox_width, bbox_height])
    transfer_normalized_keypoints = normalized_keypoints - normalized_keypoints[0,:] 
    return torch.tensor(transfer_normalized_keypoints)
    
    
def animal_pose_collection(real_num=20,syn_num=20):
    ##############collect animal_pose for real image first###################
    real_name = 'hicodet_' + str(real_num)
    syn_sd_name = 'sd_' + str(syn_num)
    syn_type_name = 'type2_' + str(syn_num)
    syn_final_name = 'final_' + str(syn_num)
    real_img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/'
    sd_img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_sd_name}/'
    type_img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_type_name}/'
    final_img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_final_name}/'

    idx_list = animal_idx
    pose_all = {}

    for idx in idx_list.keys():
        real_img_dir_pth = os.path.join(real_img_root_pth,str(idx))
        files = [f for f in os.listdir(real_img_dir_pth) if os.path.isfile(os.path.join(real_img_dir_pth, f))]
        file_seed_names = [os.path.basename(file) for file in files]
        pose_all_info_idx = {} 
        for name in file_seed_names:
            img_pth = os.path.join(real_img_dir_pth,name)
            keypoint,score,visible,bbox= cal_animal_pose(img_pth)
            h,w = get_img_hw(img_pth)
            ratio_keypoint = get_keypoint_ratio_using_bbox_animal(keypoint,bbox)
            key = name
            info ={}
            info['ratio_keypoint'] = ratio_keypoint.tolist()
            info['keypoint'] = keypoint.tolist()
            info['score'] = score.tolist()
            info['visible'] = visible.tolist()
            info['box'] = bbox.tolist()
            pose_all_info_idx[key] = info
        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/animal_{idx}.json','w') as file:
            json.dump(pose_all_info_idx,file)
        pose_all[idx] = pose_all_info_idx 
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{real_name}/animal_all.json','w') as f:
        json.dump(pose_all,f)
    #####################syn data#################    
    print('finish real statics, begin with synthetic images...\n')
    sd_all = {}
    type_all = {}
    final_all = {}
    for idx in idx_list.keys():
        sd_img_dir_pth = os.path.join(sd_img_root_pth,str(idx))
        type_img_dir_pth = os.path.join(type_img_root_pth,str(idx))
        final_img_dir_pth = os.path.join(final_img_root_pth,str(idx))
        files = [f for f in os.listdir(sd_img_dir_pth) if os.path.isfile(os.path.join(sd_img_dir_pth, f))]
        file_seed_names = [os.path.basename(file) for file in files]
        sd_pose_all_info_idx = {} 
        type_pose_all_info_idx = {} 
        final_pose_all_info_idx = {} 
        for name in file_seed_names:
            sd_img_pth = os.path.join(sd_img_dir_pth,name)
            type_img_pth = os.path.join(type_img_dir_pth,name)
            final_img_pth = os.path.join(final_img_dir_pth,name)
            if os.path.getsize(final_img_pth) < 1024*10 or os.path.getsize(type_img_pth) < 1024*10:
                continue
            sd_keypoint,sd_score,sd_visible, sd_box= cal_animal_pose(sd_img_pth)
            type_keypoint, type_score, type_visible, type_box  =cal_animal_pose(type_img_pth)
            final_keypoint,final_score,final_visible , final_box= cal_animal_pose(final_img_pth)

            sd_ratio_keypoint = get_keypoint_ratio_using_bbox_animal(sd_keypoint,sd_box)
            type_ratio_keypoint = get_keypoint_ratio_using_bbox_animal(type_keypoint,type_box)
            final_ratio_keypoint = get_keypoint_ratio_using_bbox_animal(final_keypoint,final_box)
            key = name
            sd_info ={}
            sd_info['keypoint'] = sd_keypoint.tolist()
            sd_info['score'] = sd_score.tolist()
            sd_info['visible'] = sd_visible.tolist()
            sd_info['box'] = sd_box.tolist()
            sd_info['ratio_keypoint'] = sd_ratio_keypoint.tolist()
            type_info ={}
            type_info['keypoint'] = type_keypoint.tolist()
            type_info['score'] = type_score.tolist()
            type_info['visible'] = type_visible.tolist()
            type_info['ratio_keypoint'] = type_ratio_keypoint.tolist()
            type_info['box'] = type_box.tolist()
            final_info ={}
            final_info['keypoint'] = final_keypoint.tolist()
            final_info['score'] = final_score.tolist()
            final_info['visible'] = final_visible.tolist()
            final_info['ratio_keypoint'] = final_ratio_keypoint.tolist()
            final_info['box'] = final_box.tolist()
            sd_pose_all_info_idx[key] = sd_info
            type_pose_all_info_idx[key] = type_info
            final_pose_all_info_idx[key] = final_info
        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_sd_name}/animal_{idx}.json','w') as file:
            json.dump(sd_pose_all_info_idx,file)
        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_type_name}/animal_{idx}.json','w') as file:
            json.dump(type_pose_all_info_idx,file)
        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_final_name}/animal_{idx}.json','w') as file:
            json.dump(final_pose_all_info_idx,file)
        sd_all[idx] = sd_pose_all_info_idx 
        type_all[idx] = type_pose_all_info_idx
        final_all[idx] = final_pose_all_info_idx
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_sd_name}/animal_all.json','w') as f:
        json.dump(sd_all,f)
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_type_name}/animal_all.json','w') as f:
        json.dump(type_all,f)
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_final_name}/animal_all.json','w') as f:
        json.dump(final_all,f)

def hand_animal_pose_collection(real_num=20,syn_num=20):
   
    syn_hand_name = 'hand_' + str(syn_num)
    
    hand_img_root_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_hand_name}/'

    idx_list = animal_idx
    pose_all = {}

    hand_all = {}
    for idx in idx_list.keys():

        hand_img_dir_pth = os.path.join(hand_img_root_pth,str(idx))

        files = [f for f in os.listdir(hand_img_dir_pth) if os.path.isfile(os.path.join(hand_img_dir_pth, f))]
        file_seed_names = [os.path.basename(file) for file in files]
        hand_pose_all_info_idx = {} 
        for name in file_seed_names:
            hand_img_pth = os.path.join(hand_img_dir_pth,name)
            if os.path.getsize(hand_img_pth) < 1024*10 :
                continue
            hand_keypoint,hand_score,hand_visible, hand_box= cal_animal_pose(hand_img_pth)
            hand_ratio_keypoint = get_keypoint_ratio_using_bbox_animal(hand_keypoint,hand_box)
            key = name
            hand_info ={}
            hand_info['keypoint'] = hand_keypoint.tolist()
            hand_info['score'] = hand_score.tolist()
            hand_info['visible'] = hand_visible.tolist()
            hand_info['box'] = hand_box.tolist()
            hand_info['ratio_keypoint'] = hand_ratio_keypoint.tolist()
            
            hand_pose_all_info_idx[key] = hand_info

        with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_hand_name}/animal_{idx}.json','w') as file:
            json.dump(hand_pose_all_info_idx,file)
        hand_all[idx] = hand_pose_all_info_idx 
    with open(f'/scratch/yangdejie/xz/ReVersion/hoi/animal/{syn_hand_name}/animal_all.json','w') as f:
        json.dump(hand_all,f)

def collect_json_info_for_single_hoi(json_raw_info):
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
        if 'body_dis' not in values.keys(): 
            continue
        body_ratio.append(values['body_ratio'][0][0])
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

            human_hand_keypoint +=values['hand_pose']
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
               
def compute_hodd_real_syn(data_path,syn_type,hoi_type,real_num,syn_num):
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    real_name = f'hicodet_{real_num}'
    syn_name = syn_type
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
        real_json_pth = f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{real_name}/pose_result_{hoi_idx}.json'
        with open(real_json_pth,'r') as f:
            real_info = json.load(f)

        syn_json_pth = osp.join(syn_root,f'{hoi_idx}/hodd.json')
        with open(syn_json_pth,'r') as f:
            syn_info = json.load(f)
        real_pose_info = collect_json_info_for_single_hoi(real_info)
        syn_pose_info = collect_json_info_for_single_hoi(syn_info)
        hoi_idx_collection = {}
        try:
            body_pose_dis = wasserstein_distance(real_pose_info['human_body_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['human_body_keypoint'].reshape(-1,2).cpu().numpy().flatten())
            hand_pose_dis = wasserstein_distance(real_pose_info['human_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['human_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten())
        except:
            continue
        try:
            ratio_body_pose_dis = wasserstein_distance(real_pose_info['ratio_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['ratio_keypoint'].reshape(-1,2).cpu().numpy().flatten())
            ratio_hand_pose_dis = wasserstein_distance(real_pose_info['ratio_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten(),syn_pose_info['ratio_hand_keypoint'].reshape(-1,2).cpu().numpy().flatten())
            hoi_idx_collection['ratio_keypoint_dis']  =ratio_body_pose_dis
            hoi_idx_collection['ratio_hand_keypoint_dis'] = ratio_hand_pose_dis
        except:
            hoi_idx_collection['ratio_keypoint_dis']  =torch.zeros((0,2))
            hoi_idx_collection['ratio_hand_keypoint_dis'] = torch.zeros((0,2))
        real_body_ratio_mean = torch.mean(real_pose_info['body_ratio'])
        real_hand_ratio_mean =  torch.mean(real_pose_info['hand_ratio'])
        syn_body_ratio_mean = torch.mean(syn_pose_info['body_ratio'])
        syn_hand_ratio_mean = torch.mean(syn_pose_info['hand_ratio'])
        
        real_body_distance_mean = torch.mean(real_pose_info['body_distance'])
        real_hand_distance_mean = torch.mean(real_pose_info['hand_distance'])
        syn_body_distance_mean = torch.mean(syn_pose_info['body_distance'])
        syn_hand_distance_mean = torch.mean(syn_pose_info['hand_distance'])
        hoi_idx_collection['body_pose_dis'] = body_pose_dis
        hoi_idx_collection['hand_pose_dis'] = hand_pose_dis

        hoi_idx_collection['real_body_ratio_mean'] = real_body_ratio_mean.item()
        hoi_idx_collection['real_hand_ratio_mean'] = real_hand_ratio_mean.item()
        hoi_idx_collection['syn_body_ratio_mean'] = syn_body_ratio_mean.item()
        hoi_idx_collection['syn_hand_ratio_mean'] = syn_hand_ratio_mean.item()
        hoi_idx_collection['real_body_distance_mean'] = real_body_distance_mean.item()
        hoi_idx_collection['real_hand_distance_mean'] = real_hand_distance_mean.item()
        hoi_idx_collection['syn_body_distance_mean'] = syn_body_distance_mean.item()
        hoi_idx_collection['syn_hand_distance_mean'] = syn_hand_distance_mean.item()
        body_dis.append(body_pose_dis)
        hand_dis.append(hand_pose_dis)
        
        ratio_body_dis.append(ratio_body_pose_dis)
        ratio_hand_dis.append(ratio_hand_pose_dis)
        real_body_ratio.append(real_body_ratio_mean.item())
        real_hand_ratio.append(real_hand_ratio_mean.item())
        syn_body_ratio.append(syn_body_ratio_mean.item())
        syn_hand_ratio.append(syn_hand_ratio_mean.item())
        real_body_pixel_distance.append(real_body_distance_mean.item())
        real_hand_pixel_distance.append(real_hand_distance_mean.item())
        syn_body_pixel_distance.append(syn_body_distance_mean.item())
        syn_hand_pixel_distance.append(syn_hand_distance_mean.item())

        all_hoi_collection[hoi_idx] = hoi_idx_collection

    print(f'Final average for this {hoi_type} of {syn_type} is as follows\n')
    print(f'body_hodd  average is {sum(body_dis)/(len(body_dis)*512)}.\n')
    print(f'hand_dis hodd is {sum(hand_dis)/(len(hand_dis)*512)}.\n')

        
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str,default='connect')
parser.add_argument('--resize',default=False, action='store_true')
parser.add_argument('--generated_image',default=True,action='store_true')
parser.add_argument('--model',type=str,default='sd')
parser.add_argument('--animal_pose',default=False,action='store_true')
parser.add_argument('--real_num',type=int, default=20)
parser.add_argument('--syn_num',type=int, default=20)
parser.add_argument('--data_path',type=str,default='/scratch/yangdejie/xz/ReVersion/clean_hoi')
args = parser.parse_args()

pose_dis_cal_for_generated_imgs(type =args.type,model='sd',num = args.syn_num,data_path  =args.data_path)
pose_dis_cal_for_generated_imgs(type =args.type,model='pia',num = args.syn_num,data_path  =args.data_path)
pose_dis_cal_for_imgs(type = args.type,if_resize = args.resize,num = args.real_num)
compute_hodd_real_syn(data_path = args.data_path,syn_type='sd',hoi_type=args.type,real_num=200,syn_num=50)
compute_hodd_real_syn(data_path = args.data_path,syn_type='pia',hoi_type=args.type,real_num=200,syn_num=50)