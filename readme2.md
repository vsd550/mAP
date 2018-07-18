The link for the DOTA dataset that is being used for this demo :  [link](https://captain-whu.github.io/DOTA/dataset.html) and the corresponding [paper](https://arxiv.org/abs/1711.10398). 
The **annotation format** etc. can be checked from the DOTA website.       
## Including a custom dataset in Detectron

- Convert the dataset to COCO format and place them annotations/images folders appropriately as described in the coco readme
- Add the dataset info to `lib/datasets/dataset_catalog.py`
- Edit the `train_net_step.py` file to include args for the custom dataset
- **Note:** If N is the number of classes cfg.MODEL.NUM_CLASSES will be N+1 (1 for the background class)
- Modify the config file, say `e2e_mask_rcnn_R-101-FPN_1x_d2s.yaml` to your specifications.




## Data Preparation
Download the repo **pytorch.detectron**. Let's call it the root repo.     
Create a data folder under the root repo,

```
 cd {root_repo}
 mkdir data
```

**Custom_Dataset**:Make sure to put the files as the following structure:
```
  custom_dataset
  ├── annotations
  │   ├── instances_train2018.json
  │   ├── instances_val2018.json
  │   ├── ...
  |
  └── images
      ├── train2018
      ├──val2018
      ├── ...
```     
The json files `instances_train2018.json` or `instances_val2018.json` will be created using the steps mentioned in the **coco directory**.
The images will be contained in train2018 and val2018 folders. You can give any names to the folders, till you make sure that the correct paths are added in `dataset_catalog.py`.     

Feel free to put the dataset at any place you want, and then soft link the dataset under the `data/` folder:

   ```
   ln -s path/to/custom_dataset data/custom_dataset
   ```
### Pretrained Model

We use ImageNet pretrained weights from Caffe for the backbone networks.

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1), [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l), [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)       


Download them and put them into the `{root_repo}/data/pretrained_model`        
ImageNet pretrained weights from Detectron for the backbone networks can also be used 
#### ImageNet Pretrained Model provided by Detectron

- [R-50.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl)
- [R-101.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-101.pkl)
- [R-50-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl)
- [R-101-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl)
- [X-101-32x8d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl)
- [X-101-64x4d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl)
- [X-152-32x8d-IN5k.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl)

We need to change the corresponding line in model config file(.yaml) as follows:
```
RESNETS:
  IMAGENET_PRETRAINED_WEIGHTS: 'data/pretrained_model/R-50.pkl'
```

## Training   
**Note :** This code is supported by pytorch version 0.3.1 only. So ensure that you are using that version.     
### Configuration files
The config files (.yaml files) are present in `configs/baselines`. Some insights in the config file :

- `IMAGENET_PRETRAINED_WEIGHTS:` change the imagenet pretrained weights you want to use
- `NUM_GPUS` : the number of GPUs to use
- `MAX_ITER:` : how many iterations do we want to train
- `STEPS:` : At what iteration_steps should we decay the learning rate by a factor of 10

**DO NOT CHANGE anything in the provided config files(configs/\*\*/xxxx.yml) unless you know what you are doing**

Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use.

### Adapative config adjustment

#### Let's define some terms first

 batch_size:            `NUM_GPUS` x `TRAIN.IMS_PER_BATCH`  
 effective_batch_size:  batch_size x `iter_size`  

Following config options will be adjusted **automatically** according to actual training setups:
  1. number of GPUs `NUM_GPUS`
  2. batch size per GPU `TRAIN.IMS_PER_BATCH`
  3. update period `iter_size`


- `SOLVER.BASE_LR`: adjust directly propotional to the change of batch_size.
- `SOLVER.STEPS`, `SOLVER.MAX_ITER`: adjust inversely propotional to the change of effective_batch_size.

### Train from scratch
Use `train_net_step.py`   
- `SOLVER.LR_POLICY: steps_with_decay` is supported    
-  Training warm up as in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) is supported   

Take mask-rcnn with res50-FPN backbone for example.   
  ```python tools/train_net_step.py --dataset custom_dataset --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml  --bs 8 --nw 4```   
Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use, just before the python command . So it looks like :    
  ```CUDA_VISIBLE_DEVICES=1 python tools/train_net_step.py --dataset custom_dataset --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml  --bs 8 --nw 4```   
**Note :** Ensure that you use the configuration file corresponding to the backbone you want to use.     
Use `--bs` to overwrite the default batch size to a proper value that fits into your GPUs. Simliar for `--nw`, number of data loader threads defaults to 4 in config.py.     

**Note :** The checkpoints are saved in **Outputs** folder in the root directory that is created during training    
`iter_size` defaults to 1.      

      

### Resume training with the same dataset and batch size      
```python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint} --resume```     
When resume the training, **step count** and **optimizer state** will also be restored from the checkpoint. For SGD optimizer, optimizer state contains the momentum for each trainable parameter.

**NOTE**: `--resume` is not yet supported for `--load_detectron`    

### Show command line help messages    
```python train_net_step.py --help```    
## Running Inference      
### Visualize the training results on images       
Download the pretrained model from the [model zoo of detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). Following links :  
* [R-50-C4-1x.pkl](https://s3-us-west-2.amazonaws.com/detectron/35858791/12_2017_baselines/e2e_mask_rcnn_R-50-C4_1x.yaml.01_45_57.ZgkA7hPB/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)    
* [R-50-FPN-1x.pkl](https://s3-us-west-2.amazonaws.com/detectron/35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)     
* [R-101-FPN-1x.pkl](https://s3-us-west-2.amazonaws.com/detectron/35861795/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_1x.yaml.02_31_37.KqyEK4tT/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)       
*    
Many other pretrained models can be downloaded from the above model zoo link.**NOTE** that these models are trained on the **COCO** dataset.      

Alternatively,You can train the model on your own custom dataset following the training steps outlined above and the training process automatically saves checkpoints(You can define after how many iterations, you want to save checkpoints in the `lib/core/config.py` file). Then these checkpoints can be used for inference using the below command        
```python tools/infer_simple.py --dataset custom_dataset --cfg cfgs/baselines/e2e_mask_rcnn_R-50-C4.yml --load_ckpt {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}```      

`--output_dir` defaults to `infer_outputs`, `--image_dir` contains all the images on which you want to run inference      
`--load_ckpt` : to load the saved checkpoint(.pth) file      

### Evaluate the training results - mAP metrics     
For example, test mask-rcnn on the validation set    
```python tools/test_net.py --dataset coco2017 --cfg config/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml --load_ckpt {path/to/your/checkpoint}```    
If multiple gpus are available, add `--multi-gpu-testing`.     
In my case, multi-gpu-testing was giving error, so I just used 1 GPU      
```CUDA_VISIBLE_DEVICES=0 python3 tools/test_net.py --dataset DOTA --cfg configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml --load_ckpt {} --output_dir output_dir```     

Specify a different output directry, use `--output_dir {...}`. Defaults to `{the/parent/dir/of/checkpoint}/test` 

This will save the detection results, i.e `bounding box` and `segmentation` detections in a json file and then performs evaluation using the `json dataset evaluator` and the best we have achieved for bounding boxes is:
```
 41.4 :mAP
 60.1 :AP for plane
 31.3 :AP for small-vehicle
 38.9 :AP for ship
 43.3 :AP for storage-tank
 52.7 :AP for large-vehicle
 22.0 :AP for bridge
 ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.654
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.060
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.545
```
## Supported Network modules     

- Backbone:
  - ResNet:
    `ResNet50_conv4_body`,`ResNet50_conv5_body`,
    `ResNet101_Conv4_Body`,`ResNet101_Conv5_Body`,
    `ResNet152_Conv5_Body`
  - ResNeXt:
    `[fpn_]ResNet101_Conv4_Body`,`[fpn_]ResNet101_Conv5_Body`, `[fpn_]ResNet152_Conv5_Body`
  - FPN:
    `fpn_ResNet50_conv5_body`,`fpn_ResNet50_conv5_P2only_body`,
    `fpn_ResNet101_conv5_body`,`fpn_ResNet101_conv5_P2only_body`,`fpn_ResNet152_conv5_body`,`fpn_ResNet152_conv5_P2only_body`

- Box head:
  `ResNet_roi_conv5_head`,`roi_2mlp_head`, `roi_Xconv1fc_head`, `roi_Xconv1fc_gn_head`

- Mask head:
  `mask_rcnn_fcn_head_v0upshare`,`mask_rcnn_fcn_head_v0up`, `mask_rcnn_fcn_head_v1up`, `mask_rcnn_fcn_head_v1up4convs`, `mask_rcnn_fcn_head_v1up4convs_gn`

- Keypoints head:
  `roi_pose_head_v1convX`

**NOTE**: the naming is similar to the one used in Detectron. Just remove any prepending `add_`.



## Configuration Options

Architecture specific configuration files are put under configs. The general configuration file `lib/core/config.py` **has almost all the options with same default values as in Detectron's**, so it's effortless to transform the architecture specific configs from Detectron.

**Some options from Detectron are not used** because the corresponding functionalities are not implemented yet. For example, data augmentation on testing.

### Extra options
- `MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = True`:  Whether to load ImageNet pretrained weights.
  - `RESNETS.IMAGENET_PRETRAINED_WEIGHTS = ''`: Path to pretrained residual network weights. If start with `'/'`, then it is treated as a absolute path. Otherwise, treat as a relative path to `ROOT_DIR`.
- `TRAIN.ASPECT_CROPPING = False`, `TRAIN.ASPECT_HI = 2`, `TRAIN.ASPECT_LO = 0.5`: Options for aspect cropping to restrict image aspect ratio range.
- `RPN.OUT_DIM_AS_IN_DIM = True`, `RPN.OUT_DIM = 512`, `RPN.CLS_ACTIVATION = 'sigmoid'`: Official implement of RPN has same input and output feature channels and use sigmoid as the activation function for fg/bg class prediction. In [jwyang's implementation](https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/rpn/rpn.py#L28), it fix output channel number to 512 and use softmax as activation function.

### How to transform configuration files from Detectron

1. Remove `MODEL.NUM_CLASSES`. It will be set according to the dataset specified by `--dataset`.
2. Remove `TRAIN.WEIGHTS`, `TRAIN.DATASETS` and `TEST.DATASETS`
3. For module type options (e.g `MODEL.CONV_BODY`, `FAST_RCNN.ROI_BOX_HEAD` ...), remove `add_` in the string if exists.
4. If want to load ImageNet pretrained weights for the model, add `RESNETS.IMAGENET_PRETRAINED_WEIGHTS` pointing to the pretrained weight file. If not, set `MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS` to `False`.
5. [Optional] Delete `OUTPUT_DIR: .` at the last line
6. Do **NOT** change the option `NUM_GPUS` in the config file. It's used to infer the original batch size for training, and learning rate will be linearly scaled according to batch size change. Proper learning rate adjustment is important for training with different batch size.
7. For group normalization baselines, add `RESNETS.USE_GN: True`.

## My nn.DataParallel

- **Keep certain keyword inputs on cpu**
  Official DataParallel will broadcast all the input Variables to GPUs. However, many rpn related computations are done in CPU, and it's unnecessary to put those related inputs on GPUs.
- **Allow Different blob size for different GPU**
  To save gpu memory, images are padded seperately for each gpu.
- **Work with returned value of dictionary type**

