# PIA-HOI:Pose and Interaction Aware Human Object Interaction Image Generation

## Dependencies

### 1. conda package install
`conda install -r requirements.txt`
### 2. Other tools
install detectron2 following [Detectron2](https://github.com/facebookresearch/detectron2)

install mmpose following [mmpose](https://github.com/open-mmlab/mmpose)

### 3. Specifically, for evaluation, please follow [ADA-CM](https://github.com/ltttpku/ADA-CM?tab=readme-ov-file) to prepare another enviroment for HOIF calculation.
## Download pre-trained Diffusion weight 
Download from [SD-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Inference
### Single Iteration inference
```
python pia_inference.py --prompt 'a photo of a person riding a horse' --seed 42
```
### Itertive Refinement inference
```
python iir_inference.py --hoi_path './kite/loop/round_0.png' --prompt 'a photo of a person holding a kite' 
```

## Benchmark Creation
### 1.Download HICO-DET 
follow instructions in [HICO-DET](https://github.com/fredzzhang/hicodet) to download the raw dataset,
and the dataset should be organized as
```
hicodet                            
   |   |- hico_20160224_det        
   |       |- annotations
   |       |- images
```
### 2. Generate Dataset for evaluation
We provide the config files in ./hoi_configs for dataset generation, please run
```
cd hoi_configs
python collect_images.py --data_type train --hoi_type connect --num 200 --data_path DATA_PATH
python collect_images.py --data_type train --hoi_type human --num 200 --data_path DATA_PATH
python collect_images.py --data_type train --hoi_type animal --num 200 --data_path DATA_PATH
```
Change the root image_root_pth, config_pth and data_path to your own path, and the generated dataset should be organized as
```
dataset
   |-----|-connect
   |        |-sd
   |        |-pia
   |-----|-human
   |        |-sd
   |        |-pia
```
## Evaluation
### 1. sample images 
For image sampling, use following instrcutions:
```
python samply.py --num NUMBER --type connect --data_path DATA_PATH
python samply.py --num NUMBER --type human --data_path DATA_PATH
python samply.py --num NUMBER --type animal --data_path DATA_PATH
```
### 2. utilizing following different instructions for evaluation
For following evaluation, change the data_path to your data path accordingly

For FID,KID evaluation
```
python fid.py --distance_type fid --hoi_type connect --syn_model sd
python fid.py --distance_type fid --hoi_type connect --syn_model pia
python fid.py --distance_type fid --hoi_type animal --syn_model sd
python fid.py --distance_type fid --hoi_type animal --syn_model pia
python fid.py --distance_type fid --hoi_type human --syn_model sd
python fid.py --distance_type fid --hoi_type human --syn_model pia
python fid.py --distance_type kid --hoi_type connect --syn_model sd
python fid.py --distance_type kid --hoi_type connect --syn_model pia
python fid.py --distance_type kid --hoi_type animal --syn_model sd
python fid.py --distance_type kid --hoi_type animal --syn_model pia
python fid.py --distance_type kid --hoi_type human --syn_model sd
python fid.py --distance_type kid --hoi_type human --syn_model pia
```
|     | Scenario| FID  | KID(10^-2) |
|  ----  | ----  | ----| ---- |
| SD  | H-O |  76.44  | 2.878    |
| PIA | H-O |   73.15   | 2.716  |
| SD  | H-A | 56.67   | 1.950    |
| PIA | H-A |  53.67  | 1.801    |
| SD  | H-H | 137.58   |  3.544   |
| PIA | H-H |  135.48  |  3.323   |

For HOI Metric evaluation
```
#metric pcs
python pcs.py --hoi_type connect --compare_keypoint
python pcs.py --hoi_type animal --compare_keypoint
python pcs.py --hoi_type human --compare_keypoint

#metric pdd 
python pcs.py --hoi_type connect --pos_dis
python pcs.py --hoi_type animal --pos_dis
python pcs.py --hoi_type human --pos_dis

#metric hodd
python hodd.py --type connect
python hodd.py --type animal 
python hodd.py --type human 

#metric racc
python racc.py --hoi_type connect --model sd
python racc.py --hoi_type connect --model pia
python racc.py --hoi_type human --model sd
python racc.py --hoi_type human --model pia
python racc.py --hoi_type animal --model sd
python racc.py --hoi_type animal --model pia
```
|     | Scenario| PCS(Body)  | PDD(Body) | HODD(Body) |HODD(Hand)| R-Acc A@5 |
|  ----  | ----  | ----| ---- |  ---- | ----| ----|
| SD  | H-O |  52.95  | 2.878    |16.88 | 13.40 |99.39|
| PIA | H-O |   56.15   | 2.716  |16.82 | 13.03|99.48|
| SD  | H-A | 63.15   | 1.950    |    13.37  |  16.90    |  62.22   |
| PIA | H-A |  66.28  | 1.801    |    13.51  |   16.65   |   62.43  |
| SD  | H-H | 59.47   |  3.544   |12.20|12.92|    99.2  |
| PIA | H-H |  62.16  |  3.323   |10.96|12.42|     99.6 |

For HOIF evaluation, alter to the environment [pocket](https://github.com/fredzzhang/pocket) or follow the instructions in [ADA-CM](https://github.com/ltttpku/ADA-CM?tab=readme-ov-file) to install, then run the instructions
```
conda activate pocket
git clone https://github.com/ltttpku/ADA-CM.git
mv inference2.py ./ADA-CM
cd ADA-CM
python inference2.py --syn_model sd --hoi_type connect --data_root DATA_PATH
python inference2.py --syn_model pia --hoi_type connect --data_root DATA_PATH
```
For more performance such as ANIMAL PCS and HAND PCS, you can alter the correposnding detector(like hand detector, animal 2D detector) in [mmpose](https://github.com/open-mmlab/mmpose) for evaluation.

## Acknowledgement
We gratefully thank the [diffusers](https://github.com/huggingface/diffusers), [ADA-CM](https://github.com/ltttpku/ADA-CM?tab=readme-ov-file),  [mmpose](https://github.com/open-mmlab/mmpose) and [Detectron2](https://github.com/facebookresearch/detectron2) for open-sourcing their code.
