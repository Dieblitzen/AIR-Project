# Running Image Segmentation Models

Once you have run the commands to create an Image Segmentation dataset as detailed in the previous README, you can train models as described here.

## Dependencies and Setup
1. We use `tensorflow-2`, so you should install and update tensorflow as detailed in these [instructions](https://www.tensorflow.org/install)
2. If you are using a machine with GPUs to train the model, then you should have the `cuDNN` SDK and `CUDA` toolkit installed. See [this article](https://www.tensorflow.org/install/gpu#software_requirements) for more details. If you have trouble setting those up, then these sources might help:
   * Installing `cuDNN` or [the `cuDNN` version is wrong](https://github.com/tensorflow/tensorflow/issues/23715). Run this:  
   `conda install -c anaconda cudnn`
   * [Installing `CUDA`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Tensorflow [guide](https://www.tensorflow.org/install/gpu#install_cuda_with_apt), Stack Overflow [guide](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04)
   * [You need to remove an old `CUDA` toolkit](https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one)
3. We use `tensorboard` to monitor training. This should be installed when you install `tensorflow`. Here is the [documentation](https://www.tensorflow.org/tensorboard/get_started) to help you understand it better. To visualise tensorboards created on a remote server on your machine, refer to this [StackOverflow post](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server).
4. We use `json` for the model configuration files, so you should familiarise yourself with its format.

## Setting up the Dataset

You should reference the main README file to set up the image segmentation dataset. In particular, make sure that in your `data_path...` folder you have an `im_seg` subdirectory created. Within the `im_seg` subdirectory, you should have 4 directories: `train`, `test`, `val` and `out`.

## Training a model

The main script to train models is in `ImSeg/train.py`. To train a model, you should use the following command:
```
python ImSeg/train.py --data_path [directory name] --classes_path [path/to/classes.json] --config ImSeg/configs/yourConfig.json
```
Each aspect of the script is explained below:
* `--data_path`: This is the name of your directory that contains your dataset.
* `--classes_path`: This is the path to the `.json` file that contains exactly the classes (or keys) that we want labelled info for (the same as the `--classes` argument in the `DataPipeline.py`).
* `--config`: This is the path to your `.json` model configuration file that specifies the type of model, and some of the training parameters you want to use. See below for a detailed explanation of config files.

### Saved Weights and Metrics

During training, the model will periodically save its weights if the validation IoU (intersection over union) has improved compared to the best IoU so far. Metrics for precision, recall, and IoU are logged every epoch and saved to the tensorboard and log file. The weights for the model are saved in the following directory:  
`.../data_path_[your_area]/im_seg/out/[model_name]/checkpoints/`  
Metrics are stored in the following directory:  
`.../data_path_[your_area]/im_seg/out/[model_name]/metrics/`


### Training on Server
If you are training a model, we recommend that you use a machine with GPUs. If your machine has multiple GPUs, then you can run the following before running the training command to use another GPU (eg: gpu 1):  
`export CUDA_VISIBLE_DEVICES=1`  
To make all `CUDA` devices visible again:  
`unset CUDA_VISIBLE_DEVICES`  

You also want to make sure that when you logout, the training job still runs. To do that, modify the training command as follows:
```
nohup python ImSeg/train.py --data_path [directory name] --classes_path [path/to/classes.json] --config ImSeg/configs/yourConfig.json &
```
`nohup` runs the job in the background and stores the standard output in a log called `nohup.out`. You can also use `tmux` to achieve the same.

To set up the tensorboards, on the remote machine, first run:
```
tensorboard --logdir data_path_eg/im_seg/out/[model name]/metrics/
```
This should give you a port number which is being used by the remote machine to run tensorboard. Then, to view the tensorboards locally, use 
```
ssh -N -L localhost:[your port eg: 5001]:localhost:[server port eg: 6001] gpu1
```
Consult this post for more [tensorboard remote usage details](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server).

## Model Config 

This section describes the layout of the model configuration files used to describe image segmentation models. We use `.json` files to describe the model configurations, and examples can be found int `ImSet/configs/...`. The strcture of the configuration files are described here:
```
{  
   type: type of model [eg: "RefineNet"],  
   name: unique model name [eg: "refine_net_test"],  
   backbone: model architecture backbone [eg: "resnet50"],  
   backbone_kwargs: keyword arguments for chosen backbone architecture [eg: {  
      "include_top": false,  
      "weights": "imagenet"  
    }],
   pretrained: Whether to use a pretrained Tensorflow backbone. [eg: True]  
   refine_net_blocks: List of layer names that specify the output position of different feature maps in backbone model. [eg:  
      [  
         ["layer4", "layer3"],  
         ["layer2", "layer1"]  
      ]  
   ],  
   input_shape: Shape of input image in h x w x c [eg: [224, 224, 3]],  
   classes: The specific classes you want the model to train on. Don't include this if you want the model to train on all classes defined in classes.json [eg: ["building:other"]],  
   refine_net_kwargs: Keyword arguments for the RefineNet model. Set this to an empyt dictionary if you want the default keyword arguments. [eg:   
      {   
         "reduce_channel_scale": 4,  
         "rcu_kwargs": {},  
         "mrf_kwargs": {},  
         "crp_kwargs": {}  
      }  
    ],  
    augment: Keyword arguments for data augmentation. Set this to an empty dictionary if you don't want any augmentation. [eg:   
      {  
         "rotate_range":30,  
         "flip":true,   
         "channel_shift_range":50,   
         "multiplier":1,   
         "seed":0  
      }  
    ],  
    epochs: Number of training epochs [eg: 200],  
    batch_size: Number of images per batch fed into model [eg; 16],  
    loss: Tensorflow Keras loss name (should match one of their losses) [eg: "BinaryCrossentropy"],  
    loss_kwargs: Keyword arguments for loss object. [eg: {"from_logits":true}],  
    optimizer: Tensorflow Keras optimizer name (should match one of their optimizers) [eg: "Adam"],  
    optimizer_kwargs: Keyowrd arguments for optimizer object [eg: {"learning_rate":0.0001}]  
}  
```

## Inference
After training, you can run the model on the validation or test sets to generate output segmentation maps. 