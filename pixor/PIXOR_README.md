
# Preprocessing:


The `train_labels_preprocessing.py` file contains the script to generate numpy files that will be used in normalization.  Each time that a new dataset is being used, this script must be run. 

To run use: 
```
python train_labels_preprocessing.py --data_path [path to dataset folder]
```

The generated files are: 

* `train_mean`: mean of bounding box values
* `train_std`: standard deviation of bounding box values
* `mean`: mean of images
* `std`: standard deviation of images


# Training 

The `network.py` file is used to train the model. To train run: 

```
`python network.py --gpu [how to run] --num_epochs [number of epochs to train] --batch_size [training batch size] --logfile_name [file to save logging info] --tile_size [size of tile images] --data_path [path to data folder] --num_classes [number of building classes]`
```

Each aspect of the above script is explained below:
* `--gpu`: default=0.  Set to 1 if you want to run with GPU, and set to 0 of you want to run with CPU
* `--num_epochs`: default = 300.  This is the number of epochs that the model will train over. 
* `--batch_size`: default=32.  This is the batch size during training. 
* `--logfile_name`: default='PIXOR_logfile.  This is the name of the file that you want to save logging info to. 
* `--tile_size`: default=224.  This is the size of the tile images.  It is the same number used when running the script located in `DataPipeline.py`. 
* `--data_path`: default='data_path'.  This is the name of the dataset folder that contains the pixor subfolder.
* `--num_classes`: default = 6.  This is the number of building classes. 


