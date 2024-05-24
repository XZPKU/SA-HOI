# PIA-HOI:Pose and Interaction Aware Human Object Interaction Image Generation

## Dependencies

### 1. conda package install
`conda install -r requirements.txt`
### 2. install detectron2 following [Detectron2](https://github.com/facebookresearch/detectron2)

### 3. install mmpose following [Mmpose](https://github.com/open-mmlab/mmpose)

### 4. Specifically, for evaluation, please follow [ADA-CM](https://github.com/ltttpku/ADA-CM?tab=readme-ov-file) to prepare another enviroment for HOIF calculation.
## Download pre-trained Diffusion weight 
Download from [SD-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Inference
### Single Iteration inference
```
python pia_inference.py --prompt 'a photo of a person riding a horse' --seed 42
```
### Itertive Refinement inference
```
python iir_inference.py --hoi_path --prompt
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
We provide the config files for dataset generation, please run
```

```
## Evaluation
### 1. sample images 
```

```
### 2. utilizing following different instrctions for evaluation

```

```

