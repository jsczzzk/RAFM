# RAFM: Recurrent Atrous Feature Modulation for Accurate Monocular Depth Estimating (SPL)
This is the official implementation for testing depth estimation using the model proposed in 
>RAFM: Recurrent Atrous Feature Modulation for Accurate Monocular Depth Estimating


RAFM can estimate a depth map from a single image.

https://drive.google.com/drive/folders/1EAw7EhCAgEAGH4krHmF3ObZubwmYiQvs?usp=sharing

## Pretrained Models
We have updated all the results as follows:
[models]([https://drive.google.com/drive/folders/1IhUsEEY-oKfgcsTX2uHuENMe7u-1Pzik?usp=sharing](https://drive.google.com/drive/folders/1EAw7EhCAgEAGH4krHmF3ObZubwmYiQvs?usp=sharing))

## KITTI Evaluation
You can predict scaled disparity for a single image used RAFM with:
```shell
python test_simple.py --image_path='path_to_image' --model_path='path_to_model' 
