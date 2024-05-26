from hoi_configs.animal import animal_idx
from hoi_configs.connect import connect_idx
from hoi_configs.human import human_idx
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from skimage import io
# from skimage.transform import resize
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor
from torch.nn.functional import adaptive_avg_pool2d
from cleanfid import fid
def calculate_activation_statistics(images, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    act_values = []

    with torch.no_grad():
        for img in images:
            img = ToTensor()(img).unsqueeze(0).to(device)
            act = model(img)[0].detach().cpu()
            act = adaptive_avg_pool2d(act, output_size=(1, 1))
            act_values.append(act.view(act.size(0), -1))

    act_values = torch.cat(act_values, dim=0)
    mu = torch.mean(act_values, dim=0)
    sigma = torch.cov(act_values, rowvar=False)

    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = torch.sqrtm(sigma1 @ sigma2, True)
    if torch.iscomplex(covmean):
        covmean = covmean.real

    return torch.norm(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)

def frechet_inception_distance(real_images, generated_images, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    mu_real, sigma_real = calculate_activation_statistics(real_images, model)
    mu_fake, sigma_fake = calculate_activation_statistics(generated_images, model)

    fid_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score.item()

def calculate_fid(images1, images2, batch_size=64):

    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    

    model = torch.nn.Sequential(*list(model.children())[:-2])
    

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])


    features1 = extract_features(images1, model, transform, batch_size)


    features2 = extract_features(images2, model, transform, batch_size)

    mean1, cov1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mean2, cov2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    cov_mean_sqrt = sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_mean_sqrt):
        cov_mean_sqrt = cov_mean_sqrt.real

    fid = np.sum((mean1 - mean2) ** 2) + np.trace(cov1 + cov2 - 2 * cov_mean_sqrt)

    return fid

def extract_features(images, model, transform, batch_size):
    dataset = ImageDataset(images, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []

    with torch.no_grad():
        for batch in dataloader:
            if batch.dim() == 4:
                batch = batch.squeeze(2)
            features_batch = model(batch).squeeze(-1).squeeze(-1)
            features.append(features_batch.cpu().numpy())

    features = np.concatenate(features, axis=0)

    return features

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        return image

def get_files(pth):
    all_pths = []
    for filepath,dirnames,filenames in os.walk(pth):
        for filename in filenames:
            img_full_name = os.path.join(filepath,filename)
            if os.path.getsize(img_full_name) < 1024*2:
                continue
            all_pths.append(os.path.join(filepath,filename))
    return all_pths


def collect_syn_pth_caption(hoi_type,syn_model,syn_num):
    img_pths = []
    
    img_root = f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_model}_{syn_num}_caption/'
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    else:
        hoi_idxs = animal_idx
    for hoi_idx in hoi_idxs.keys():
        img_dir_root = os.path.join(img_root,str(hoi_idx))
        hoi_idx_imgs = get_files(img_dir_root)
        img_pths.append(hoi_idx_imgs)
    return [element for sublist in img_pths for element in sublist]

def seperate_hoi_class_fid(hoi_type,syn_model):
    real_root = f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/hicodet/'
    syn_root = f'/scratch/yangdejie/xz/ReVersion/hoi/{hoi_type}/{syn_model}/'
    if hoi_type=='connect':
        hoi_idxs = connect_idx
    else:
        hoi_idxs = animal_idx
    fids = []
    for hoi_idx in hoi_idxs.keys():
        real_img_dir_root = os.path.join(real_root,str(hoi_idx))
        syn_img_dir_root = os.path.join(syn_root,str(hoi_idx))
        try:
            fid_score = fid.compute_fid(real_img_dir_root,syn_img_dir_root)
            fids.append(fid_score)
            print(f'FID for {hoi_type} is {fid_score}.\n')
            with open(f'/scratch/yangdejie/xz/ReVersion/fid_outputs/output_{hoi_type}_{syn_model}.txt','a') as f:
                f.write(f'FID score for {hoi_type} for model {syn_model} is {fid_score}.\n')
                f.close()
        except:
            continue
    with open(f'/scratch/yangdejie/xz/ReVersion/fid_outputs/output_{hoi_type}_{syn_model}.txt','a') as f:
            f.write(f'Average FID score for all {hoi_type} classes for model {syn_model} is {sum(fids)/len(fids)}.\n')
            f.close()
            
            
def all_compute_fid(real_img,syn_img,hoi_type,syn_model):
    feat_model = fid.build_feature_extractor("clean", "cuda:0", use_dataparallel=True)
    real_np_feats = fid.get_files_features(real_img,model=feat_model)
    mu1 = np.mean(real_np_feats, axis=0)
    sigma1 = np.cov(real_np_feats, rowvar=False)
    
    syn_np_feats = fid.get_files_features(syn_img,model=feat_model)
    mu2 = np.mean(syn_np_feats, axis=0)
    sigma2 = np.cov(syn_np_feats, rowvar=False)
    fid_score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'FID for {hoi_type} is {fid_score} for {syn_model}.\n')
    
def all_compute_kid(real_img,syn_img,hoi_type,syn_model):
    feat_model = fid.build_feature_extractor("clean", "cuda:0", use_dataparallel=True)
   
    real_np_feats = fid.get_files_features(real_img,model=feat_model)
    mu1 = np.mean(real_np_feats, axis=0)
    sigma1 = np.cov(real_np_feats, rowvar=False)
    
    syn_np_feats = fid.get_files_features(syn_img,model=feat_model)
    mu2 = np.mean(syn_np_feats, axis=0)
    sigma2 = np.cov(syn_np_feats, rowvar=False)
    #fid_score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    kid_score = fid.kernel_distance(real_np_feats,syn_np_feats)
    print(f'KID for {hoi_type} is {kid_score}.\n')
       
       
def collect_real_pth(data_root,hoi_type,real_num):
    img_pths = []

    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    data_path = osp.join(data_root,f'{hoi_type}/hicodet_200')
    for hoi_idx in hoi_idxs.keys():
        img_dir_root = os.path.join(data_path,str(hoi_idx))
        hoi_idx_imgs = get_files(img_dir_root)
        img_pths.append(hoi_idx_imgs)
    return [element for sublist in img_pths for element in sublist]

def collect_syn_pth(data_root,hoi_type,syn_model,syn_num):
    img_pths = []

    if hoi_type=='connect':
        hoi_idxs = connect_idx
    elif hoi_type=='animal':
        hoi_idxs = animal_idx
    else:
        hoi_idxs = human_idx
    data_path = osp.join(data_root,f'{hoi_type}/{syn_model}')
    for hoi_idx in hoi_idxs.keys():
        img_dir_root = os.path.join(data_path,str(hoi_idx))
        hoi_idx_imgs = get_files(img_dir_root)
        img_pths.append(hoi_idx_imgs)
    return [element for sublist in img_pths for element in sublist if element.endswith('.png')] 
       
parser = argparse.ArgumentParser()
parser.add_argument('--distance_type',default='fid')
parser.add_argument('--syn_model',type=str,default='pia')
parser.add_argument('--hoi_type',type=str,default='connect')
parser.add_argument('--real_num',type=int,default=100)
parser.add_argument('--syn_num',type=int,default=50)
parser.add_argument('--data_root',default='/scratch/yangdejie/xz/ReVersion/clean_hoi')
parser.add_argument('--real_data_root',default='/scratch/yangdejie/xz/ReVersion/hoi')
parser.add_argument('--if_crop_real',default=False,action='store_true')
parser.add_argument('--if_caption',default=False,action='store_true')
args = parser.parse_args()

real_img_pth = collect_real_pth(args.real_data_root,args.hoi_type,args.real_num)
syn_img_pth = collect_syn_pth(args.data_root,args.hoi_type,args.syn_model,args.syn_num)
if args.if_caption:
    syn_img_pth = collect_syn_pth_caption(args.hoi_type,args.syn_model,args.syn_num)
origin_root = '/scratch/yangdejie/xz/ReVersion/hoi/hicodet/hico_20160224_det/images/train2015/'
origin_real_pth = [origin_root + path.rsplit('/', 1)[-1]   for path in real_img_pth]

real_images = [io.imread(path) for path in real_img_pth]
syn_images = [io.imread(path) for path in syn_img_pth if path.endswith('.png')]


if args.distance_type=='fid':
    if args.if_crop_real:
        all_compute_fid(origin_real_pth,syn_img_pth,hoi_type=args.hoi_type,syn_model=args.syn_model)
    else:
        all_compute_fid(real_img_pth,syn_img_pth,hoi_type=args.hoi_type,syn_model=args.syn_model)
        
        
if args.distance_type=='kid':
    if args.if_crop_real:
        all_compute_kid(origin_real_pth,syn_img_pth,hoi_type=args.hoi_type,syn_model=args.syn_model)
    else:
        all_compute_kid(real_img_pth,syn_img_pth,hoi_type=args.hoi_type,syn_model=args.syn_model)