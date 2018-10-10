# Literature Review

**List of Reviewed Articles**

1.  Small Object Detection in Optical Remote Sensing Images via Modified Faster R-CNN
2.  R-CNN for Small Object Detection
3.  Geospatial Object Detection in High Resolution Satellite Images Based on Multi-Scale Convolutional Neural Network

### [Geospatial Object Detection in High Resolution Satellite Images Based on Multi-Scale Convolutional Neural Network](https://www.mdpi.com/2072-4292/10/1/131/htm)

#### Summary

This paper describes a multi-scale convolutional neural network for detecting objects of all sizes in high resolution satellite (HRS) images. While several effective methods have been used for bounding box object detection in the past, none are particularly well-suited for detecting small objects within HRS images. This architecture is R-CNN inspired, with a few key improvements given below

- Shared Multi-Scale Base Network
- Multi-Scale Object Proposal Network
- Multi-Scale Object Detection Network

**Shared Multi-Scale Base Network**

One major problem Faster R-CNNs have in detecting objects of different scales has to do with their base feature maps. Traditionally, Faster R-CNN feature maps do not have strong semantic representations for objects of all scales. However, in order to be able to detect small objects in large images, feature maps with strong semantic representations for a variety of differently-sized objects is required. To produce feature maps with strong semantic representations on all scales, a shared multi-scale base network was used.

This shared multi-scale base network is built from a "bottom up" section and a "top down" section. The bottom up architecture is a VGG-16 net, and it is used to produce several initial feature maps with different levels of semantic representation. In the top down part of the architecture, feature maps are taken from several places in the middle of the bottom up portion and combined (using element-wise addition and deconvolutional layers) with other feature maps produced at the end and in the middle of the VGG-16 network. The multi-scale base network outputs several differently-sized feature maps with receptive fields of 8, 16, 32, and 64 pixels with respect to the original image.
![Multi-scale base network image](https://www.mdpi.com/remotesensing/remotesensing-10-00131/article_deploy/html/images/remotesensing-10-00131-g003.png)

**Multi-Scale Object Proposal Network**  
To detect multi-scale objects in images, a region proposal network (RPN) is used on each feature map. The RPN takes feature maps as inputs and proposes bounding boxes of 2 scales and 5 aspect ratios for each feature map, outputting an "objectness" score and the coordinates of each bounding box. Because there will be so many more bounding boxes that do not surround objects than ones that do, a classifier is also trained to supress a large portion of the negative samples. This classifier, which is composed of convolutional layers followed by softmax, gives proposal scores for each bounding box. Boxes with scores below 0.5 are discarded. ![Multi-scale object proposal network](https://www.mdpi.com/remotesensing/remotesensing-10-00131/article_deploy/html/images/remotesensing-10-00131-g005.png)

**Multi-Scale Object Detection Network**  
After generating feature maps and object proposals, the next step is to determine what objects are in the object proposal bounding boxes, if any, and adjust the bounding boxes to more closely fit any objects that might be inside of them. In this paper, two 3×3 convolutional layers were applied to the feature maps to obtain classification and bounding box regression results. While different layers could be used in this step, such as residual or inception units, this paper simply used 3x3 convolutional layers for simplicity. The architecture was trained and tested on the VHR-10 dataset and achieved a 89.6% mAP value.

#### Pros

- Accurate, well suited for detecting smaller objects as well as bigger ones
- Intuitive

#### Cons

- Slow
- Not specialized enough for detecting smaller objects, might waste a lot of resources unless we revise it to be more specific.
