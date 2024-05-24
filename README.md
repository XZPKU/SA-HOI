# PIA-HOI:Pose and Interaction Aware Human Object Interaction Image Generation

## Dependencies

### 1. conda package install
`conda install -r requirements.txt`
### 2. Other tools
install detectron2 following [Detectron2](https://github.com/facebookresearch/detectron2)

install mmpose following [Mmpose](https://github.com/open-mmlab/mmpose)

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
### 2. utilizing following different instrctions for evaluation

```

```
|  表头   | 表头  |
|  ----  | ----  |
| 单元格  | 单元格 |
| 单元格  | 单元格 |

For other performance like animal PCS or JOINT PCS, you can alter the correposnding detector in [Mmpose](https://github.com/open-mmlab/mmpose) for evaluation.

## Acknowledgement
We gratefully thank the [diffusers](https://github.com/huggingface/diffusers), [ADA-CM](https://github.com/ltttpku/ADA-CM?tab=readme-ov-file),  [Mmpose](https://github.com/open-mmlab/mmpose) and [Detectron2](https://github.com/facebookresearch/detectron2) for open-sourcing their code.
