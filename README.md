# SpatioGS: Spatiotemporal-Aware Density Control for Dynamic Scene Rendering with Gaussian Splatting

## Data Preparation

The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1). [Neu3D Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites.

Meanwhile, We adopted [4DGS](https://github.com/hustvl/4DGaussians) scripts for processing the Neu3D and HyperNerf dataset.

## Download
Another way to download the source code of SpatioGS:
Link: https://pan.baidu.com/s/10jbx7iTwVebjFYJbxlw02Q?pwd=utmb code: utmb

## Training

For training Neu3D scene such as `cut_roasted_beef`, run
```python
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```

For training multiple Neu3D scenes, run
```python
bash train_n3d.sh
```

For training Hypernerf scene such as `broom`,run
```python
python train.py -s  data/hypernerf/broom/ --port 6017 --expname "hypernerf/broom" --configs arguments/hypernerf/broom.py 
```

For training multiple Hypernerf scenes, run
```python
bash train_hyper.sh
```

## Rendering

Run the following script to render the images.
```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
```

## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```
