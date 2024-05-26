import torch
import os
import numpy as np
import diffusers
import random
import pickle
import math
import os.path as osp
import json
from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
from hoi_configs.human import human_idx
from Visualizer.visualizer import get_local
get_local.activate()
from PIL import Image
from tqdm.auto import tqdm
#from ..pose_api import call_pose_api
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers import StableDiffusionPanoramaPipeline
from mystablepipe import StableDiffusionSAGPipeline
from pia_hoi_pipe import StableDiffusionPIAPipeline
from diffusers import DDIMScheduler
import random
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from torchvision import transforms
from utils import CrossAttnStoreProcessor, view_images
# from pose_api import call_pose_api
from pose_attention import generate_attention_map, generate_hand_attention_map
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
import argparse
inferencer = MMPoseInferencer(pose2d='human')
def call_pose_api(img_path):
    
    #result_generator = inferencer(img_path,shows = False,return_datasamples=False)
    result_generator = inferencer(img_path,shows = False,return_datasamples=True)
    #result = next(result_generator)
    result = [result for result in result_generator]
    return result

pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        variant="fp16"
        ).to('cuda')

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)



pipe2 = StableDiffusionPIAPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        variant="fp16"
        ).to('cuda')

pipe2.scheduler = DDIMScheduler.from_config(pipe2.scheduler.config)
pipe2.scheduler.set_timesteps(50)
bsz=1
sp_sz = pipe2.unet.sample_size
prompt = 'a photo of a human riding a clock'


def sample_img(hoi_type,hoi_idx,prompt,num,data_path):
    collect_num=0
    seed_list = []
    sd_data_root = osp.join(data_path,f'{hoi_type}/sd/{hoi_idx}')
    pia_data_root = osp.join(data_path,f'{hoi_type}/pia/{hoi_idx}')
    os.makedirs(sd_data_root,exist_ok=True)
    os.makedirs(pia_data_root,exist_ok=True)
    while(collect_num < num):
        if len( os.listdir(sd_data_root))>=num:
            break
        seed = random.randint(1,10000)
        #seed=3156
        latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(seed)).to('cuda') 
        image = pipe(prompt, latents=latents).images[0]
        sd_save_path = osp.join(data_path,f'{hoi_type}/sd/{hoi_idx}/{seed}.png')
        pia_save_path = osp.join(data_path,f'{hoi_type}/pia/{hoi_idx}/{seed}.png')
        image.save(sd_save_path)
        pia_image = pipe2.pia_forward(prompt, latents=latents,pose_scale=2,inter_scale=0.5,refine_image_path=sd_save_path).images[0]
        pia_image.save(pia_save_path)
        seed_list.append(seed)
        collect_num+=1
    return seed_list        

def sample_image(seed,hoi_type,hoi_idx,prompt,data_path):

    sd_data_root = osp.join(data_path,f'{hoi_type}/sd/{hoi_idx}')
    pia_data_root = osp.join(data_path,f'{hoi_type}/pia/{hoi_idx}')
    os.makedirs(sd_data_root,exist_ok=True)
    os.makedirs(pia_data_root,exist_ok=True)
    
    latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(seed)).to('cuda') 
    image = pipe(prompt, latents=latents).images[0]
    sd_save_path = osp.join(data_path,f'{hoi_type}/sd/{hoi_idx}/{seed}.png')
    pia_save_path = osp.join(data_path,f'{hoi_type}/pia/{hoi_idx}/{seed}.png')
    image.save(sd_save_path)
    pia_image = pipe2.pia_forward(prompt, latents=latents,pose_scale=2,inter_scale=0.5,refine_image_path=sd_save_path).images[0]
    pia_image.save(pia_save_path)
    return

parser =  argparse.ArgumentParser()
parser.add_argument('--type',type=str,default='animal')
parser.add_argument('--num',type=int,default=50)
parser.add_argument('--data_path',type=str,default='/scratch/yangdejie/xz/ReVersion/clean_hoi')
parser.add_argument('--seed',type=bool,default=True)
args = parser.parse_args()
if args.type=='animal':
    configs = animal_idx
elif args.type == 'connect':
    configs = connect_idx
else:
    configs = human_idx

hoi_type = args.type
all_seeds = {}
if args.seed:
    with open(f'./dataset/{hoi_type}_seeds.json' ,'r') as f:
        all_seeds = json.load(f)
    for hoi_idx in configs.keys():
        prompt = configs[hoi_idx]
        seed_idx = all_seeds[hoi_idx]
        for seed in seed_idx:
            sample_image(seed,hoi_type,hoi_idx,prompt,data_path=args.data_path)
        print(f'finish generating images for {hoi_idx} : {prompt}.\n')

else:
    all_seeds = {}
    for hoi_idx in configs.keys():
        prompt = configs[hoi_idx]
        seed_list_for_hoi_idx = sample_img(hoi_type,hoi_idx,prompt,args.num,args.data_path)
        all_seeds[hoi_idx] = seed_list_for_hoi_idx
        print(f'finish generating images for {hoi_idx} : {prompt}.\n')
        with open(osp.join(args.data_path,f'{hoi_type}_seeds.json') ,'a') as f:
            json.dumps(all_seeds)