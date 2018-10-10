# Literature Review

**List of Reviewed Articles**

1.  DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images (https://arxiv.org/pdf/1805.06561.pdf)
2.  Learning Aerial Image Segmentation from Online Maps (https://arxiv.org/pdf/1707.06879.pdf)
3.  Faster R-CNN (https://arxiv.org/pdf/1506.01497.pdf)
4.  YOLO (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
5.  Small Object Detection in Optical Remote Sensing Images via Modified Faster R-CNN  (https://www.mdpi.com/2076-3417/8/5/813)
6.  R-CNN for Small Object Detection (https://www.merl.com/publications/docs/TR2016-144.pdf)
7.  Geospatial Object Detection in High Resolution Satellite Images Based on Multi-Scale Convolutional Neural Network    (https://www.mdpi.com/2072-4292/10/1/131/htm)

### [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

#### Summary 

YOLO approaches object detection as a regression problem. It is a fast and relatively simple network that is used primarily for real-time object detection. YOLO tends to be less accurate and make more localization errors than state of the art techniques, but it is faster, better at generalizing, and less prone to detect false positives than other top architectures.

**Design**  
YOLO first divides the input image into an SxS grid, where each grid cell is responsible for detecting objects whose centers lie in that cell. Each grid cell predicts B bounding boxes and confidence scores for each box. The confidence scores  reflect the "objectness" of the box and the predicted accuracy of the bounding box. Each grid cell also predicts C conditional class probabilities. By multiplying this and the original confidence scores together, class-specific confidence scores can be assigned to each grid cell.  

The architecture of this network is inspired by the GoogLeNet model. 24 convolutional layers for feature extraction are followed by 2 fully connected layers for outputting coordinates and object probabilities. ![YOLO Architecture](https://cdn-images-1.medium.com/max/1600/1*ZbmrsQJW-Lp72C5KoTnzUg.jpeg)


#### Pros

- Fast
- YOLO is a single network, whereas other methods have separate parts that need to be trained and optimized separately
- Learns very general representations of objects (better at generalizing than R-CNN, good for geographically different areas?)
- Sees entire image during training and test time so it's better able to encode contextual information about classes. Because of this, YOLO detects fewer false positives than R-CNNs

#### Cons

- More localization errors than other methods, accuracy worse than state of the art methods. This is because loss is the same for large vs small bounding boxes and while small errors for large boxes are insignificant, small errors in localizing small boxes can be significant. 
- Has an especially difficult time with small objects
- The number of objects YOLO can detect is limited

### [Small Object Detection in Optical Remote Sensing Images via Modified Faster R-CNN](https://www.mdpi.com/2076-3417/8/5/813)

#### Summary

The paper claims to improve object detection for small objects by making the following five modifications to the standard Faster R-CNN architecture:
1.  Feature map fusion before anchor boxes are proposed
2.  Shrinking anchor box sizes to match dataset statistics
3.  Network module to incorporate context
4.  Random rotations for data pre-processing
5.  Balanced sampling to combat class imbalance

#### Design

Feature map fusion before anchor boxes are proposed:
Draws inspiration from feature pyramid networks.  In a normal scenario, there is a feature map upon which one would propose anchor boxes.  That feature map would be the last layer of some pre-trained network backbone (e.g. ResNet-50).  This paper proposes that we take the last few layers of that pre-trained backbone and fuse them together through addition (and the necessary 1x1 convolutions to reduce the dimensions so that we can add feature maps of different dimensions).  One would then propose anchor boxes on this fused feature map.  The idea is that we can combine high-resolution information with semantically meaningful features, which is important for accurately capturing features for extremely small objects.

Shrinking anchor box sizes to match dataset statistics:
The anchor box sizes for standard object detection benchmarks are designed to catch large objects; this paper’s response is to choose better anchor box sizes based on the range of object sizes in the specific problem that the algorithm is facing.  This experimentally improves accuracy.

Network module to incorporate context:
Idea is that context can help with object recognition.  Essentially, for each anchor box proposal they also extract a context proposal which encapsulates the anchor box proposal.  Both of these proposals are fed through a combination of ROI layers and some fully connected layers to get the final classifications and coordinates.

Random rotations for data processing:
Empirically show that their “RR” method improves accuracy.  However, unclear how it is different than standard rotations in data augmentation. 

Balanced sampling to combat class imbalance:
Irrelevant to our problem for now because we do not expect class imbalance.

### [R-CNN for Small Object Detection](https://www.merl.com/publications/docs/TR2016-144.pdf)

The paper claims to improve object detection for small objects by making the following 2 modifications to the standard R-CNN architecture:
1.  Shrink anchor box sizes
2.  Context module

Shrink anchor box sizes:  Same reasoning as above paper.

Context module: For each proposal, also include a context region which is essentially a larger region enclosing proposal.  Feed both regions into network, and at some point later in the network, concatenate the two different feature maps together.



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
