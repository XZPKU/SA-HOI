from animal import animal_idx
from connect import connect_idx
from human import human_idx
import json
from PIL import Image
import os
import torch
import argparse
import os.path as osp
import torchvision.transforms.functional as F
image_root_pth = './hicodet/hico_20160224_det/images/train2015/'
config_pth = './hicodet/hico_20160224_det/annotations/trainval_hico.json'
with open(config_pth, 'r') as f:
    anns = json.load(f)
    
def crop_and_store(ann, image_name,idx,collect_num,data_path):
    for item in ann['hoi_annotation']:
        if item['hoi_category_id'] == idx:
            sub_idx = item['subject_id']
            obj_idx = item['object_id']
            sub_bbox = torch.Tensor(ann['annotations'][sub_idx]['bbox'])
            obj_bbox = torch.Tensor(ann['annotations'][obj_idx]['bbox'])
            x_min = torch.min(sub_bbox[0],obj_bbox[0]).item()
            y_min = torch.min(sub_bbox[1],obj_bbox[1]).item()
            x_max = torch.max(sub_bbox[2],obj_bbox[2]).item()
            y_max = torch.max(sub_bbox[3],obj_bbox[3]).item()
            img_pth = os.path.join(image_root_pth,image_name)
            origin_image = Image.open(img_pth)
            cropped_image = F.crop(origin_image, y_min, x_min, y_max - y_min, x_max - x_min)
            resized_img = cropped_image.resize((512,512))
            os.makedirs(osp.join(data_path,f'animal/hicodet_{collect_num}/{idx}/'),exist_ok=True)
            save_idx_path = osp.join(data_path,f'animal/hicodet_{collect_num}/{idx}/')

            cropped_image.save(os.path.join(save_idx_path,image_name))

            
def crop_and_store_connect(ann, image_name,idx,collect_num,data_path):
    for item in ann['hoi_annotation']:
        if item['hoi_category_id'] == idx:
            sub_idx = item['subject_id']
            obj_idx = item['object_id']
            sub_bbox = torch.Tensor(ann['annotations'][sub_idx]['bbox'])
            obj_bbox = torch.Tensor(ann['annotations'][obj_idx]['bbox'])
            x_min = torch.min(sub_bbox[0],obj_bbox[0]).item()
            y_min = torch.min(sub_bbox[1],obj_bbox[1]).item()
            x_max = torch.max(sub_bbox[2],obj_bbox[2]).item()
            y_max = torch.max(sub_bbox[3],obj_bbox[3]).item()
            img_pth = os.path.join(image_root_pth,image_name)
            origin_image = Image.open(img_pth)
            cropped_image = F.crop(origin_image, y_min, x_min, y_max - y_min, x_max - x_min)
            resized_img = cropped_image.resize((512,512))
            os.makedirs(osp.join(data_path,f'connect/hicodet_{collect_num}/{idx}/'),exist_ok=True)
            save_idx_path = osp.join(data_path,f'connect/hicodet_{collect_num}/{idx}/')
            cropped_image.save(os.path.join(save_idx_path,image_name))

         
def crop_and_store_human(ann, image_name,idx,collect_num,data_path):
    for item in ann['hoi_annotation']:
        if item['hoi_category_id'] == idx:
            sub_idx = item['subject_id']
            obj_idx = item['object_id']
            sub_bbox = torch.Tensor(ann['annotations'][sub_idx]['bbox'])
            obj_bbox = torch.Tensor(ann['annotations'][obj_idx]['bbox'])
            x_min = torch.min(sub_bbox[0],obj_bbox[0]).item()
            y_min = torch.min(sub_bbox[1],obj_bbox[1]).item()
            x_max = torch.max(sub_bbox[2],obj_bbox[2]).item()
            y_max = torch.max(sub_bbox[3],obj_bbox[3]).item()
            img_pth = os.path.join(image_root_pth,image_name)
            origin_image = Image.open(img_pth)
            cropped_image = F.crop(origin_image, y_min, x_min, y_max - y_min, x_max - x_min)
            resized_img = cropped_image.resize((512,512))
            os.makedirs(osp.join(data_path,f'human/hicodet_{collect_num}/{idx}/'),exist_ok=True)
            save_idx_path = osp.join(data_path,f'human/hicodet_{collect_num}/{idx}/')
            cropped_image.save(os.path.join(save_idx_path,image_name))   
def acquire_image_names(idx,data_path,collect_type ='animal',collect_num=100):
    num = 0
    image_names = []
    for ann in anns:
        if num>collect_num:
            break
        exist_hois = [ item['hoi_category_id'] for item in ann['hoi_annotation']]
        if idx not in exist_hois:
            continue
        else:
            image_names.append(ann['file_name'])
            if collect_type =='animal':
                crop_and_store(ann,ann['file_name'],idx,collect_num,data_path)
            elif collect_type == 'connect':
                crop_and_store_connect(ann,ann['file_name'],idx,collect_num,data_path)
            else:
                crop_and_store_human(ann,ann['file_name'],idx,collect_num,data_path)
            num+=1
    return image_names




parser  = argparse.ArgumentParser()
parser.add_argument('--hoi_type',type=str,default='connect')
parser.add_argument('--data_type',type=str,default='train')
parser.add_argument('--num',help='image nums for each hoi class',type=int)
parser.add_argument('--data_path',help='store path',type=int)
args = parser.parse_args()

if args.hoi_type=='animal':
    hoi_idxs = animal_idx
elif args.hoi_type=='connect':
    hoi_idxs = connect_idx
else:
    hoi_idxs = human_idx
          
img_names_dict = {}
for key in hoi_idxs.keys():
    text_prompt = hoi_idxs[key]
    print("Collecting images for {}.\n".format(text_prompt))
    img_names = acquire_image_names(key,data_path=args.data_path,collect_type =args.hoi_type,collect_num = args.num)
    img_names_dict[key] = img_names
with open(osp.join(args.data_path,f'{args.hoi_type}_{args.num}.json'),'w') as f:       
    json.dump(img_names_dict, f, indent=2)

