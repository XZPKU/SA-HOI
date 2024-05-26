from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
import json
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
# image_root_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/images/train2015/'
# config_pth = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/annotations/trainval_hico.json'
image_root_pth = './hicodet/hico_20160224_det/images/train2015/'
config_pth = './hicodet/hico_20160224_det/annotations/trainval_hico.json'  # alter to your dataset path
from pose_api import call_pose_api, call_hand_pose_api
import cv2
import numpy as np
from scipy.spatial import distance
from scipy import ndimage
import argparse
import PIL
from PIL import Image
import numpy as np
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

def pairwise_distances(A, B):

    A = A.unsqueeze(1)  
    B = B.unsqueeze(0)  
    diff = A - B  
    distances = torch.norm(diff, dim=2) 
    return distances

def min_k_distances_and_indices(distances, k):

    flattened_distances = distances.flatten()  
    sorted_indices = torch.argsort(flattened_distances)  
    min_indices = sorted_indices[:k] 
    min_distances = flattened_distances[min_indices]  


    min_indices_A = min_indices // distances.size(1)
    min_indices_B = min_indices % distances.size(1)

    return min_distances, min_indices_A, min_indices_B


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

        concatenated_tensor = np.concatenate(contours, axis=0)
        coordinates = concatenated_tensor.reshape(-1, 2)
    else:
        coordinates = torch.zeros((1,2))
    return coordinates


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
    keypoint_list = torch.tensor(result[0].pred_instances.keypoints)[0]

    obj_keypoint = torch.Tensor(obj_corrd)
    human_keypoint = keypoint_list
    distances = pairwise_distances(obj_keypoint, human_keypoint)
    min_distances, min_indices_A, min_indices_B = min_k_distances_and_indices(distances, 3)
    points_A = obj_keypoint[min_indices_A]
    points_B = human_keypoint[min_indices_B]

    concatenated_points = torch.cat((points_A, points_B), dim=0)
    sorted_distances_indices = torch.argsort(min_distances)
    min_distances_return  =  torch.cat((min_distances[sorted_distances_indices], min_distances[sorted_distances_indices]), dim=0)

    return concatenated_points, min_distances_return

def cal_dis_between_h_o_w_human_keypoint(img_pth,human_keypoint):
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
    obj_keypoint = torch.Tensor(obj_corrd)
    distances = pairwise_distances(obj_keypoint, human_keypoint)
    min_distances, min_indices_A, min_indices_B = min_k_distances_and_indices(distances, 3)
    points_A = obj_keypoint[min_indices_A]
    points_B = human_keypoint[min_indices_B]

    concatenated_points = torch.cat((points_A, points_B), dim=0)
    sorted_distances_indices = torch.argsort(min_distances)
    min_distances_return  =  torch.cat((min_distances[sorted_distances_indices], min_distances[sorted_distances_indices]), dim=0)

    return concatenated_points, min_distances_return

def generate_inter_attention_map(joint_coordinates, distance_scores,image_size=(512,512), radius=20, scale=100, variance=1000):

    epsilon = 1e-6

    attention_map = torch.zeros((1, 1, image_size[0], image_size[1]))
    attention_maps  =[]
    for coords, score in zip(joint_coordinates, distance_scores):
        y, x = coords
        x_range = torch.arange(image_size[1]).float()
        y_range = torch.arange(image_size[0]).float()
        xx, yy = torch.meshgrid(x_range, y_range)
        mask = ((xx - x).pow(2) + (yy - y).pow(2)) <= radius**2
        center_atten = 1 / score
        attention_weight =  center_atten* torch.exp(-(torch.pow(xx - x, 2) + torch.pow(yy - y, 2)) / (2.0 * variance + epsilon))
        attention_weight *= scale
        attention_map += attention_weight.view(1, 1, image_size[0], image_size[1])
        #attention_maps.append(attention_weight.view(1, 1, image_size[0], image_size[1]))

    #attention_maps = torch.stack(attention_maps,dim=0)
    #attention_map = torch.max(attention_maps,dim=0)[0]
    attention_map /= attention_map.max()

    return attention_map
