# Scalable Automatic Mapping and Recognition

This repository provides information to build geo-spatial datasets and train models to 
automatically map regions of interest. The goal of this project is to assist automated mapping efforts for use in providing humanitarian relief after natural disasters, for example.

So far, our project has a structured pipeline to extract satellite imagery using [IBM's `PAIRS` API](https://github.com/IBM/ibmpairs), and labelled bounding boxes using `overpy`; an [OpenStreetMap](https://www.openstreetmap.org/) Python API. We then provide pipelines to furhter transform the raw data into specific formats required by a couple of object detection and semantic segmentation deep learning models. After passing the data through these trasnforms, we provide scripts to train and evaluate the model performance on the extracted datasets. Currently, training must be done on personal resources. (In the coming weeks, we will provide checkpoint files for pre-trained models that have worked well on our data.)

The overall strcuture of the project can be divided into the following components:
1. A pipleine to extract raw labelled data. 
2. A pipeline to transform the raw data into specific formats required by the following single-stage models:
   * [YOLO](https://arxiv.org/pdf/1506.02640.pdf) (deprecated)
   * [PIXOR](https://arxiv.org/pdf/1902.06326.pdf) (bounding box detection)
   * [RefineNet](https://arxiv.org/pdf/1611.06612.pdf) (semantic segmentation)
3. Scripts to train and evaluate each of the abovementioned models.


## DataPipeline

The file `DataPipeline.py` is the script to be run to extract raw, labelled data. This script creates a new (local) directory to store the labelled satellite data for a particular area.   

Note, that to run this script, one also needs another file named `ibmpairspass.txt` that must contain the following line of text:  
``` pairs.res.ibm.com:[PAIRS user email][PAIRS user password]```


To run `DataPipeline.py`, use the following:  
```
python DataPipeline.py --data_path [directory name] --query_path [path/to/query.json] --classes [path/to/classes.json] --tile_size [Integer n]
```

Each aspect of the above script is explained below:
* `--data_path`: This is simply the name of your directory that will store the the extracted data. It is advised to name your directory `data_path_regionName` (eg: `data_path_dallas`).
* `--query_path`: This is the path to a `.json` file that specifies a PAIRS query for a certain region, including the layers that should be returned (for now, must be only RGB layers). For example queries, please check `PAIRS_Queries/...`
* `--classes`: This is the path to the `.json` file that contains exactly the classes (or keys) that we want labelled info for. Each "key" or "tag" must correspond to one that is used by the [Overpass API]((https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide)). For references on how to look for tags, please check [this link](https://wiki.openstreetmap.org/wiki/Tags). The structure of this file is simply a dictionary of "super classes" (more generic keys like "building") and an associated list of "sub classes" (eg: "hospital", "parking" etc. The "other" tag is used for any label/box of a particular superclass that doesn't fit into any subclass tag). For reference, please check `classes.json`.
* `--tile_size`: This is simply an integer that specifies the size of the square tile (in pixels) that the entire area will be "cut up" into. For example, a tile size of 224 corresponds to 224 x 224 square tiles that will partition the entire area. Leftover tiles at the edges smaller than 224x224 will not be included. We only support square tiles for now.

Running the above command will generate three directories: `data_path/images`, `data_path/annotations` and `data_path/raw_data`. The `raw_data` simply contains a `.jpg` image of the entire queried area, along with a `annotations.pkl` file that contains all the raw bounding boxes (in pixels) for the entire image. More concretely, the annotations are stored as a dictionary in the following format:
```
{
  'super_class_1': 
  {
    'sub_class_1': 
    [
      [list of (pixel_x, pixel_y) nodes for label 1],
      [list of (pixel_x, pixel_y) nodes for label 2],
      ...
    ],
    'sub_class_2':
    [
      [list of (pixel_x, pixel_y) nodes for label 1],
      ...
    ],
    ...
  },
  'super_class_2':
  {
    'sub_class_1': 
    ...
  },
  ...
} 
```
The `data_path/annotations` directory contains `.json` files for annotations for each tile, in the same format as above. Note that each `annotation_i.json` file will contain pixel node coordinates with respect to the tile's frame, and not the global 'full area' frame.

Finally, `data_path/images` directory simply contains `.jpg` files for each tiled image from the entire area. Thus, `image_i.jpg` in this folder is simply the `i`'th tile.
