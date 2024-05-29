import torch
import os
import numpy as np
import diffusers
import random
import pickle
import math
import json
import argparse
from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
# from Visualizer.visualizer import get_local
# get_local.activate()
from PIL import Image
from tqdm.auto import tqdm
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

parser = argparse.ArgumentParser()
parser.add_argument('--prompt',type=str,default = 'a photo of a human riding a clock')
parser.add_argument('--save_path',type=str,default='./images/')
parser.add_argument('--seed',type=int,default=42)
args = parser.parse_args()


latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(args.seed)).to('cuda') 
prompt = args.prompt
# image = pipe(prompt).images[0]  
image = pipe(prompt, latents=latents).images[0]
image.save('./images/pia_sd.png')
pia_image = pipe2.pia_forward(prompt, latents=latents,pose_scale=3,inter_scale=1,refine_image_path='./images/pia_sd.png').images[0]
pia_image.save('./images/pia.png')