# Literature Review

**List of Reviewed Articles**

1.  DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images (https://arxiv.org/pdf/1805.06561.pdf)
2.  Learning Aerial Image Segmentation from Online Maps (https://arxiv.org/pdf/1707.06879.pdf)
3.  Faster R-CNN (https://arxiv.org/pdf/1506.01497.pdf)
4.  YOLO (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
5.  Small Object Detection in Optical Remote Sensing Images via Modified Faster R-CNN  (https://www.mdpi.com/2076-3417/8/5/813)
6.  R-CNN for Small Object Detection (https://www.merl.com/publications/docs/TR2016-144.pdf)
7.  Geospatial Object Detection in High Resolution Satellite Images Based on Multi-Scale Convolutional Neural Network    (https://www.mdpi.com/2072-4292/10/1/131/htm)
8. Using Convolutional Networks and Satellite Imagery to Identify Patterns in Urban Environments at a Large Scale (https://dl.acm.org/citation.cfm?id=3098070)

### [Faster-RCNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)

#### Summary

Attempts to reduce the computational bottleneck of region proposal methods in the RCNN architecture. Specifically, replacing selective search with a deep CNN drastically improves cost. Introduces RPNs (regional proposal networks) that share convolutional layers with object detection networks that have been showed to perform well.

**Design**
Uses the feature maps created by Fast- RCNN to generate the region proposals. Then, simultaneously regressing on the bounds of the region and the object-ness scores on each location, region proposals are predicted.
Switches between fine-tuning for object detection and for generating the region proposals, thus sharing the convolutional features between both tasks.
Generating the region proposals: slide small network over CNN feature map, produce k anchor boxes at each window. Therefore, we have 2k scores for the box-classification layer, and 4k coordinates for the box regression layer. ![Faster-RCNN Architecture](https://andrewliao11.github.io/images/faster_rcnn/faster_rcnn_netwrok.png)

#### Pros
- Translation Invariant
- Much faster, no extra cost for scale, since single image is used from RCNN architecture
- Anchor boxes replace the much slower selective search

#### Cons
- Still uses regression (linear?)
- Still has to output k anchors for every every sliding window position (discards the irrelevant one only in Loss computation)
- Not specific to satellite imagery (but possibly good for small objects) 

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

### [You Only Look Twice: Rapid Multi-Scale Object Detection in Satellite Imagery](https://arxiv.org/pdf/1805.09512.pdf)

#### Summary 
YOLT was created to address the difficulties of detecting small objects in high resolution satellite images. Often satellite images can encompass areas many square kilometers in size with hundreds of millions of pixels. Objects in these images can be tiny (10-100 pixels), which makes them very difficult for many other algorithms to detect. YOLT evaluates satellite images of arbitrary size at a rate of 0.5 square km/second. This paper claims that YOLT can localize objects only 5 pixels in size with high confidence. This paper is very vague but there is code we can look at and use. See hps://github.com/CosmiQ/yolt 

**Requirements**   
Because of the unique challenges faced when working with satellite images, object detection algorithms must take into account (1) small spatial extent of objects, (2) complete rotation invariance, (3) small amounts of training data, and (4) very high resolution inputs.    

**Architecture**  
YOLT uses a 22 layer architecture that downsamples by a factor of 16. Inputting a 416x416 grid would yield a 26x26 prediction grid. YOLT's architecture is inspired by YOLO, but optimized for small, densely packed objects. A passthrough layer (similar to identity mappings in ResNet) to preserve fine grained features, and each convolutional layer besides the last is batch normalized with leaky relu activation. The last layer predicts bounding boxes and classes. The default number of boxes per grid cell is 5.  

**Training Data**   
Training data is collected from small chips of large images. Some hyperparameters: 5 boxes per grid cell, initial learning rate of 10^-3, weight decay of 0.0005, momentum of 0.9. Training took 2-3 days on a single NVIDIA Titan X GPU. 

**Test Procedure** . 
Images of arbitrary size can be partitioned and individually run through a trained model. 


#### Pros

- Can evaluate images of arbitrary size
- Fast
- Can detect tiny objects in HRES satellite images

#### Cons

- Not optimized for building detection (maybe something we could add)

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

#### Summary

The paper claims to improve object detection for small objects by making the following 2 modifications to the standard R-CNN architecture:
1.  Shrink anchor box sizes
2.  Context module

#### Design

Shrink anchor box sizes:  Same reasoning as above paper.

Context module: For each proposal, 
also include a context region which is essentially a larger region enclosing proposal.  Feed both regions into network, and at some point later in the network, concatenate the two different feature maps together.



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

### [Using Convolutional Networks and Satellite Imagery to Identify Patterns in Urban Environments at a Large Scale](https://dl.acm.org/citation.cfm?id=3098070)

#### Summary

Addresses the problem of scalability; for models models trained in one city, how good are they in different cities? Study how different classes compare across cities.

**Design**
- Models: ResNet and VGG-16
- Features extracted from these models to perform large scale comparison of urban environments (across cities)
- Ground truth labels from Urban Atlas
- Images from Google Maps static API
- 10 cities in Europe with 10 land use classes
- Why not OSM: Because the labels are too specific (e.g.:baseball diamonds, churches, roundabouts). Also Urban Atlas is for Europe.
- Why Urban Atlas: Comprehensive and consistent, large scale, curated by experts, used over the last decade. Also the land use classes reflect higher functions (socio-economic, cultural) functions of the land as used in applications. ![Basic Outline](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxITEhMSEhMWFhUVFh0YGRgXGBwXGhcYHRgYGR4dHB8aHSggHholHhoZITIiJykrLjAuGSAzODMtNygtLisBCgoKDg0OGhAQGi0lHyUtLS0tKy0tLS0tLS8tLS0tLS0tLS0tNS0tLTUrLSstLS0tLS8tLS0rLS0tLS0tLSstLf/AABEIAKMBNQMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAABAUDBgcCAQj/xABQEAACAgEDAQYCAwkJDwMFAAABAgMRAAQSITEFBhMiQVEyYQdxgRQVI0JSkZOx0iQzNGJyc6Gz0QgWNUNUY3SCorLBwuHw8VPD0xclZIOS/8QAGAEBAQEBAQAAAAAAAAAAAAAAAAECAwT/xAAbEQEBAQEAAwEAAAAAAAAAAAAAARECEjFBA//aAAwDAQACEQMRAD8A7jjGMBjGMBjGMBlT3o1U8WnL6fZ4gdP3wkLRkUEcAnkGvld+mW2VPeo/uZ/5Uf8AWpgVPeeTVrqEig8QpqlVNy9IDHIGkNgeUyQs4BJrdGvvlKvfPX/cwleFY2O5mvTTnZth8Xwigfduu18W9o28qCazo2R9ZoYpQBLGkgBsB1DAHpYsdaOBouo7y9oyeMFjEQXw2DCCRyF8WAMb3bXV0eRqG1lCEDcbZcx7xa6SRwFMUaamJQTpZdxjM00TKfwhv4I28QV5ZBai83zGBomh7xa4JplMJ3Np42dGhmLlmgLu/ibtqhHAUow3E+oLKDsPdrW6mTxl1KoGR12siPGrK0UcnR2ayrMykg0dvQcjLrGAxjGAxjGAxjGAxjGAxjGAxjGAzVpew9bcgj1O1ZHZjZYsAfyTXlFEgDkg0d5oBcP0m9ovBpo3RnX8KASjFTtKP1KkGro/YM50vb2oq/uqYCgCxnk6n0A3Dnr9d9ely1ZHTdb2JrZElH3SA7LKqEXSbyhQ0AL2BSObvr+MRmdOytVtlQz2GRglkkqS9gm1vgfP8auig5zGLtnUiAfh9QGQ0SZ3YHkAUS/Pp+bnqb8xd4dVHIUOplYO26MtI1WFplJPVT6exIrpeNMdQHZutFVqt3nQkMAKUIu5VpPV9x5vg1fAr32VodarqZtQHQDkCrY7ACT5B8T21D4doAJDHOUp2rq6aJ9XqBtrz+K97RRu9wJYEbTyL6+tZlk7zaoAEzzBWWmbc9IbG4Cm4Ivg2OL4xq47dlBqO8bIHvTuxBcKE3NZRto3Wg2lhTUu/iz0FmT3UZjpIC7MzFeSxJY8nqSST9frlqWF1Yv2ysqId5PjB0825GK/CNrGmIIJINECunU0Lo57k7xAMAYJj8IsKDy6qwrnlQGILccqQLy6a6NdfS8rOx49SCzal1tlUBE+FSrPZFi7IZL+rAgx9612qW02oBKbiAgajs37OoJbgjpV+1jLXsrtDxlZvDZKagHqyCitfH8qvsyWXFXYr3vjPWAxjGAxjMWoi3LtDMvI5XrwQfX3qvqOBlxmlTaDXR6ldvjTBSoSVpgE8IQBWEiCRQ0xlDMT4bA+IvK7fLkaXtccbVb8Eh3fgx5zsMgHPEguQKCClBbJNkhuOM1nsWbtIyp90IBGaDCkFDwI23eVyb8XetAkdfSjkSOPtHdSrIuxyhdnR1dJNWrF0G88pAG+MAiwAG5GBuOU/e3+CyfWn9ama9NL2vHFIIomdgT4e5ob3fuhuSWNxGtOvNPbNyB8ObtJ9YdPqTqQAtrsFLxWrkUfCTYMQhbn1Y/yVDc8YxgMqe1u34dO4SQPZXd5RfFke/yy2znv0jSVOlnaPCFn/WbCVejvvpaJqTj+L/1ydou8MMqF1D0CRyACaBJoXzVZx7UaqyFHlA4F/wDfX554XUSHylifxeefL/0/sxGPKuuwd7dO7Kg37mQyVtHlVb+KjxyCPr4y608wdQwujfUUeCR/wzlHYtxxah0+IkRJXI6hmI9avbfTqT750fuw96WE2D5ascg0SLH19c1Y1KtMYxmWjGMYDGMYDGMYDGMYGl/SqwGmhvn8OK9/3uTp86vOUKhjO9VDLY3IRYJv29+br1r7M6j9LyMdLBtBJ+6V6fzcvX5f9M5fqwxqQdUFMtjawBqjR9/U8Xxmb7anpZwGJ4zOhjUkjyFgWLny3VDir56UBfsIUsayRgNuUKT4bEGuBdA9GAuxybHGQnO65YiVdT5geCDdG76Vzz9h98kaXW7udqqVP72SbAHPBJ4cV62LPXCvUZLh3bZ4oILgDi1UUTfUEUt1fIzPMgoN0Rj5gOSgHW/mvtXQk3zkbUxUfFQFmJO5b5YcgqAfMCPY0OT0AoSdOALcG45eeAetmmN/PhqvhgTVnIOz9z4gui06qxYBKBJJsWa5PJ49TmHvCIVeNpjJUjogEak+ZBNJT0CShG4V68Adchdgdqw6TR6KOTxF3javkd6qzTMq0KA9TfB60Tn2LtyWcs0cEcsaTDYwKvSblTd5Safb4rc7SAVFHm9sIWq0mlYBYtZMrWUFhmUByWHRR+USjE0ByOlj3qBpSG05m1XClQV8zjdO8TMrBSb3xDn6q5bLTTSMVl26BVYAOoJVRI5u7OzhuSbo9fTMesDrLKBo43RhzUY89+FbO/Niy42BGb8HfqMCFotHoiyrHqZWLbdgJJTyeG6kAIF2klLP4xPXdm26WHYiICSFULZ6mhWaw3aDRkBdAkcj+JsHlssu0gUq8k0GNNQCjzE8DYuz9S8ilnjMZsjaTfHv0H/Y9cCVjGMBjGMD4WA65qHa/aeuTUSyQxM8KRtEqcU0oiMqy1W8rvCw8GuSfnmD6Vv3iH+e/wCRs5kF+Wce/wBfG5i46rqO2O0VkmTwoyqhQkixyNuJMILbQ/K+aawG3L4QJBBzL2drdZqC8cw8AGNWV0QqVZo1tSXai25m4UGtnJBzlG0AEsQAOv8A3/Zlfqe2Qb8Fd3ux6X7k45/W34WZ7fo7xB7j8+U/fI/uOX/U/rEz86zwbgxc7yBZHQA/IfP5533tP/BS3/6MP/t50561ltGMYzSmc1+k+ENqIrH+K/5mzpWc1+lHXBZ446JuIHgc1vYcen/HDPXpqP3OxuhY60Ov2V6Vk7u5Cpnj8wsMaDcj4bO4elqeDXXMOmYkWt9L3L6enUZPbWlYlG1d54UhQDt4Fk+/pftkjnyy96IxGETdKFKk2pBtma9zEfELD1wAaHyzofc9a0WnF35OtVfJ9PTOVTaGbcjOx3Mw8SjuNbgAqnkEKqg/nzrnduDZpokoil6N1HJ6506zG+fayxjGYbMYxgMYxgMYxgMYxgaV9KWtMUOlbbuX7qAYVfl8KX83Nc5zXW+BuZ4nDAOTag+WQjcQLJ62T+fOj/SuqmHSqwBDakCjfP4GY1wOpqueBd+mc2SEBCg5O4FT7qSPY8Hrz879szWuUHZu5U7ZEFnj/u0PIFdOavoAj309FSvxEWAoNVyPxfa+nFcdJ2o0LboiF8xIqh77gRweGBG48c8ZjkQEyxFWEiEqxFE21EkEHjctDab689MivnZYdZQgG11PWwBJTButUKrrzyec+TyAMYwVKOVIBNAbrVlP8Uk+p68EAis8dogFQbKqG3bgOhHG6/YdDz0N9LOeoGVd6SpurgupNMDtIsEgA+u7rYq/TBXbO5AP3Bpg3XwxfrRs/wBOTNf2PFKd9FJQKEsZ2yD7fxh/FYFT6g5F7mA/cWnti3k+IiieTyfnl1m2Gm9o6fVPqdNHJqDEVJ2um4R6pQyS7SA3kmHhCwbDIZK4LBJkHZvaXIbVrwBR2LTHfuNjbdBfLwefl1y87S0KzRtG90aII4ZWBtWU+jKQCD7jMPYereSOpK8WNjHJXA3r+MB6KwKuB7OMCwxjGAxjGAxjGBpP0qAeBDZoeL/yNnItR22gJSMbz8uFH1n/AL+rOr/TDoDNo02k2km+gasBGsf05xmPR0o2UQ9169PyjzXGcO5z5bTb8S10jScysTxwo4UfX7jMq6Wh8NV+Yj/x75E0GspCp6rzz8v+OXsUfjRB1raQbJYAUKBNE2eoHT1HW85dW/STWma+OnbYaFW3PQ+v2/aeufovXf4JT+Yi/UmcI7TnhjA8OpHF8kWF6mwOl37/AJs71rjfZan3gjPPB6Jno/O6kmVsuMYzopmud5u6aax1dpChVdvCg8WT6/XnzvT2rPpk8QyQRxmeJA7g+RHZFZntgvFubsAcZXQ99HSONpowS20WtoCGecB1U7nIdYg4QbmpvWrwWa+wfR+irtE7VuBNKBdc116XR+zM2q7ixu+4ysPKFoKAKBv7DnzUd+UUX4aHzuleLyHT/FHycal/xYvXnnMGr77uHCLCqgOQzM1+SteqkcABzJowKNg+IB1NgmRm0fcNIt+ydxuvaKFJZ/F5689c2XsrReDCkW4tsFbj1PJN5Qr3wAI8SMKhfZvL0BU0cJZgVpRukWuTfyyHofpBSRQ5h2jwopTcgtEkTTuZGG2hCBMR4l0TE4oVeDG64zW+yO9iz6hYBHt3RmRW37iQGK/CBYHlPLUPQXmyYV58Qe4/PjxB7j8+fn7W6TcX48wdq9/i6f2Zet2LCmkWcTAyWHRyPKWHIQIRzz1J6EfWMsms9dY7IXHuPz43j3H58/PPaWoEriR1UEChQuh9Z5Y/M/0ZgRbBoD83XM2tR+j8Z4h+FfqH6s95QxjGBqP0k9iS6uCGOJCxWcOSCo2gRyC/MRYsgVz1+0adH3V19IfuVga834SMkMbsj8J7Funvx0GdfxksWVyPSd2+0UkY/cpYVQbfF69R8fFEA3XQ0chzdy+0vF8QQX4l7yJEtSFAXguBRA28fmrnOqd5J3WB9iFtylSVajGCCN/HJA6muR+qFpCE1QXTgyI8SlyJSyx8khjuJHmBFAcn6uQ8V8nL4e53acibJNK6e1yxFRfuBIel2COeo9TfqPuP2gkrJ9ygx7Niyq0e7b6Aq0nJHpdjm+OQe34xiaqu6+iaHSwxOKZFogm65PqCf15a5T6/sPxC/wCGdFeVZCFoXtREK3V7SEFj5sPXiJL3aUKS2omIUEjzHykrIvlrmgHoAc+lm8qNjvKrQ6GVNVqZSy+FKIyqgUwcKUYsfXhUo/ZXFmsfs6KVht1E4Zl2sygqXpVj3EhQAwo89PO9DkEWU/YtyFxNIoO3yhuBtRk4vpw10OLF8nAtsZrEXY6xBBNqZnpEVhb7WO7rxZ5I559eousaXsFHA26rUGiS5bgyFlQWwZQOqBhQFcAUvlIbPjNdl7qBq3anUcCuJGA6k31689fljA2LGeZUsEWRYqxwR8x88oNJ916RAsxbWIOsirUy+9oOJFHPwU1UNrHnA8d9/wB6j/l/8pzjnbWnMMxCg+HMbocU343p69c7drdPFrolMcoKhrteeQCCCPQi+QeRlH2r9Hkc6FGmYcgghRYINgjnOHfFvWr8cdbQRLIpIBDHgH0Irr7/APjLeJgrBD+9yMOBwN4+r0YcfWBm+D6LE9dU5Ht4a1+ux9hyTJ9GsZBXx2r+QP7cxfz7WY0KPsnSo2/YFtlYE8qKIta9jRv3zrHaI/8Atvv+AT0r0X0yFqu40TBdkhQgdQtkn8rk9eTln25Ds0MiEltkNbj1NACz8zWdfy56nsufF1jGM6sqjt7tn7nMd7AriTlztG5Iy4A56nafsB9sp9T30KFS0J2ANvO5QWKQpM3hgtzQeqPJI9uc2uaFXFOoYXdMARf258k0qN8SKeb5UHnjn6+Bz8hgazP34jUSkQyN4Ra9pStiLMxYEsAeIXFC+SB71J1/bkyaxoFQeFHAk0j0DQZpxRJkXbxDwabk80Bl59yR8+RObvyjm7u+Obs/nOZPDFk0LIomuoF8H5cn85wNS0/fhWY3EQlKOoDhjLLEfKSCVtF6CwCSarMsHfEGQRvCys4VgpZF2qVjJ3MX2F/wqeUHnzVdZsa6KIVUaCulKOOCOOPYkfac9HTIatF4II8o4IFA/WB0wMXZ5m2/hwgbj4Lr4Fvr/H318q9byVjGBwbWeWR1/jH9eRdQ10PrPys9ftOWGritnU9dzbfqs8Zgi0BI3HgfrzKqwxE/9emZkTapYtQA5Pv8gMmyBRxVn0Uf8ci6rUJHbPTuBwB8K/V7n55FtkfoGH4V+ofqz3niE+VfqH6s95tkxjGAzm30r99tX2fLp003hVIjs3iIW5UqBVMPfOk5xP8AugP4Ro/5qT/eTAqP/rB2p/8Ajfom/wDkzHp/pY7RQEImkUEliFhKgk8k8SdSfXKLsXsNZoHmPjMRMsQWFQxTcjP4sl3UXlr06NzxWbLP9HkINLqWId3SJiFpmWSOMI3s5cvF/LC+hwMTfTD2pR/g36Jv/kzvfZsxeKJ26sisa9yoJz8mdowhHljBsI7qCfUKxF/0Z+nINU7RQaaEhZG06uXI+BNoW1B+N74rkL1b8VWovpEBBB6EVlb2jCsaGSpnCjkLI5bbYs1u5oc11oH6s8aHUzo8UEyJzGx3rIzklNgNhox13X1OWxPNeuQVvZ8kGoXfFIzruvh2BB5NEE2OvT6vlkmPQqPV+K6u5HHHqf8AzlFP3SXzCKZ0jklDPGK2kDqq0LBv5+3HArZbAHPGBrfb3eg6XULG8e6MputfiuyOLocV0v1Bv0y27H7Wj1MYkjur2kEUVNXR5IuiOl9cre9Pdw6toirhCm4EkWSDRHqOlH8+Zu7fYK6UMBKX3kE2AB5QRYAs+vPJ6D7QvMYxgMZoHY3ficljOiuNkTKscZiYNNqBBEAZJmWVCSbkGwLt6EsALSTv5p1KXHLyWVqCt4bIdQpDbWPG7TyAHo3FXRoLjXdjI7eLGzQzf+pHQLV0DqfLIvyYGvQg85HHa8kFDWKFH/rxgmE/NwbaH/WtR+XfGVKd99ryiaB0RWQAjaRGrQpITK6uUUebhrC8cnpe4EYBGBAINg8gjoRn3KWXsZoiX0biI3ZiYEwP7+UcxsefMlcmyrdMy6Xttd4hnQwSnhVcgpIf80/R/Xy8NXJUYFrlV3qP7j1P8036stcqO938C1X8w/8AunAt8YxgMYxgMYxgMYxgMYzHqCwRigBfadoJoFq4BPoLwONLopSz3FJ8ZryN7n5Z618GoKio2AHW42J+yhznQNd3q8HVafSyhAZfjO8+TezrDXFHeUKnkUSvW8wDv7CIhI8UtbQSVEe3+Crq3rdIDSxt6jrwLyYOT6kamisennA9WMT7m+vjp8sp9T2ZqiD+AnJo/wCKf+zO9DvlCXCLHKxaYwpQj87K0ysRcgoBoXHmo9KBzH2L3v8AulYQsLJLNAkyh2XwyG2MyqwNuyq18L6c10xjPi2SD4V+ofqzJmPTFiilwA5UbgDYDVyAfUXmTK0YxjAZxH+6CcCfR2QPwUn+8mduzBqtMrg2BdEBioYr8xeB+RBOB0ar4PNXj7oH5fz+L7c/SJ7vzxLvUxIFA3i2ltVssR4inzNQNUOpHXzGPoNFqZSsbso3RA20SqW2zKxYEL8Wxgp9OlKOSQ/Ocsq0fMOnvn6t0+gEulgG4o4iQpIvxI2wCx8vcHgjK3S93NQmxBKnhqFHwqWNTbjZZCSTF5bvr6fjZtIFcDAg6DstI9jHc0ipt3M7t1rdW5jVkD82ZO0hS79yqEtiWXcAAOvUVWS8jdoaRJY2ikFq4o11+z5jrgQNHp96lozEUflGRB09b555/WcsdPAAmwgEc9BQ6k9M0Ps3srtHSSkQruQtz5l2MLrcVLAg17c8eudCXAoe0NbHC9STQo3BopbUCa6N05Ncep647PbTyOBE8DCuURVJrqehsc7fT0GUnfLu/LJKH08AN/GwcAseAPKxAFe/r9mXHc/sb7niBZalckvfoBdLx6Ac/WT8sDYgMYxgRZOzoWG1ooyu0rRRSNpIJWq+EkA18hnxey4AVYQxWgCqdi2qi6C8cAWeB7nJeMCAnYmlAAGnhABsARoAD7jjr88n4xgMxanTJIpSRVdT1VgCD9hyHN2LCzFmD2TZqWQD8wehnj7wQe0n6aX9vAwfceo0/MDeNEP8TKx3qP8ANyt19fLJdmvOoyH292vFNoNcFtZE00peJxskTyN1U/imjTC1NWCRln94IPaT9NL+3mnd++y4Wi1IhU7tLp5JJZS7uYx4RcQruY00gALeyennU4HRcZWfeCD2k/TS/t58+8EHtJ+ml/bwLTGVf3gg9pP00v7ePvBB7Sfppf28C0xlX94IPaT9NL+3j7wQe0n6aX9vAtMZV/eCD2k/TS/t4+8EHtJ+ml/bwLTGVf3gg9pP00v7ePvBB7Sfppf28Ce+mQ3aKdxBNgGytUT7kUK9qyJqOxNM5jLwofCbcgrgNs8O9o4NLQFg1QroMx/eCD2k/TS/t4+8EHtJ+ml/bwJi6GIMXEaBi24ttFlgCAxNXuokX8znzT9nwodyRRq1VaoqmrurA6XzWRPvBB7Sfppf28feCD2k/TS/t4FpjKv7wQe0n6aX9vH3gg9pP00v7eBaXjKvQ934IpmnQP4jKEJMjsNos9GYj165aYDGU692dIS7S6eGV3dnLvEjN5mLAEkEkAEAfIDPX97Gh/yPTfoY/wBnAkds6N5YmSOQxsfUVTD1U2DQI4scjr8sr4o5ZZISokgEIKuW2nfe241sG18ikyCuOByTs+67sLs+KNpG0enpRdCCOyegA46k0B9eV69k6NZkSTQ6b8KxjULEjBWWNpST5BwRuBPHIUc3gbbjKn+9jQ/5Hpv0Mf7OP72ND/kem/Qx/s4FtjKvS9gaeKVZYYo4iFZT4capuDFTyVA6bf6ctMBjGMBjGMBjGMBkTs7tGOcO0ZJCSNG1qVp0bawpgDwR16ZLys1sjRTJJc7rJth8NFVkRiWbxW43AVwTdAVxgWeMYwGMZG7R1qQxtK97VHQCySTQVR6sSQAPUkDAi9tdoNGEiiAaea1jB5AoeaR/82gIJ9yVUcsMqu8/Z6wdka+NST+5NQWZuWdzE5ZmPqxPP9AoADLTsbROC08/79L1F2IkHwxqfl1Y+rEnpQEfv3/g3tD/AEOf+pfAvMYxgMYxgMYxgap39knUaQxeLt8c+L4QnbyeBLW77mBk279nT1q+Mrezu2desMEUkUzSNFDuZoZLNxS+IzMAArbkXg0w3ixZzZe8vby6NFkeN2UsQxUEiNQjMWegSF8tbqoXZIFnKxe+BDMrQk0dSQUdSPC0xjBY2ereIKAwIXYs+u+6o/EMqwmQhlaJ2B/cOjKgMT5FEnj8mwW3A+bN4zWH75xeM0KoSylrJdEXakWnlZtzMBwNSgr5N0rMek74jY7TR7WWWVAFYUQms+5V6n4rK3/R1AwNrxmrw98gfB/c8v4U8VTEJ4qQ7ztsAW10T8Kn1pTP7u9vrqg9I0bJW5HFMA10SCOho0wtTXBOBc4xjAYxjAYxjAqNONVHvURpIDI7BjKQdrOWArYaoGuvpniTtaZX8MxRBtu4jxjwvPJPhUB5W6+xy6yu7Q7EgmLNJGCzKqbvUKjl1o+nmN/Ohd4EPU62RwYnigO+xt+6KJrb0qO7BK8jkEj5Zi0bSq24xxPJTAE6joC29goEdDqLIFmlvoKy/wB6Wlu9jXakeduNjrIlc8BXVSB/FA6Csr+yNBCr6SQadVsuIz4jOyeR+SGXk7Qy9fxvXAtn7SnHWKEUL51FcXV/vfS88J2vKbIjgNXf7o6bWKH/ABfowK/WKzGvdDRitsZXaoVSGa1Ak8QUSb4Yn7OOgGZT3Z0xNlCfOJK3NW8TCcHr6OoNdKFYGWFdQ0yO6KiKjghZC+4sUrjYBxtPPzy0xjAYxjAYxjAYxjAYxjAYxjAZRaE/dcq6g/weMnwB6SP0M/zXqE+RLc2pHrtRjqZDpEakWjqGH5J5EI/jOPi9Qh9C6nLpFAAAFAcADoBgfco+/f8Ag3tD/Q5/6l8vMo+/f+De0P8AQ5/6l8C8xjGAxjGAxjGBi1GmRwA6K4BsBgGo+/PrnxdLGCWCLuN2dos2ADZ9bCgfYPbM2MCKOzoQAoij2qQQNi0COhArgj3z6dBDd+FHdlr2C9zVZ6dTQs+tZJxgYG0cZ2kxodhtbUeU+68cH6s9afTIl7EVbNnaoWz7mvXMuMBjGMBjGMBjGMBkLX9raeEgTTRxlhYDuq2PlZ5ybnBv7pf990P8iX/eTA65re3uz5UaNtZBTCjU6qR8wQ1g/PKZdfpHEcMuq0gihNq6TIGkroKB/B8WGr4vSgSM/KWMD9mf30aH/K9P+lT+3JOh7Z00zFIZ4pGAshHViBYF0D0sj8+firOvf3Nv8M1X+j/+4uB+hcpNHHNL4j/dMiVLIoVVioBZGUfFGT0Hvl3eU3ZesSOOQu1Dx5z0JoCVySa6KPUnjAxajejENqp6G0FgkBAZiFVaEe4sSRwAeo988LODyNfKfqji91A/xPruWvewRYyVLq9KzgtZY11RxyknF8UCr+/Iv2POFZdEVAA4ZeF2SWQQq1tq+V2gCvhoDjAxayfwwxbWzEKCTtjiatqsxHEXLUp8o5+Vc5Lg00j3t1spo0fJDwfUH8F1HtiWbSyHzAEt5SNrH4l28gCq2ybS38YC8yaPVadELpYEjlj5GBLEAkkFbAquSOle+Bp30h9uazQeB4WoZ/F33vSM1t2VW1B+Uf6MZVfTBqhLH2fKoIDrIwDCjREJFj3z5gdaxjGAyt7Z7QMeyKKjPNYjB5Ar4pG/iICCfclV6sMssq+zOyPCmnnaVpGmbgN0iQXSJ6gc2eaJ5oYEnsvQLBGI1JPUszcs7nlmY+rE/wBg4AyXmuds94mgmkTapVVgC728Nd80kq270dijwwOhstVcjKqLvfPLJGIkiCyeAVUyG2Eg1O6ztICXCCrKDYF/jUA3jMOr0ySxvFIoZJFKMp6MrAgg/IgkZo8f0mIyNINO1LpvHouAT+5hqaHlopR2bgb3A+Wucn95e8Wo00kBZECDTaieZUcsbi8EBVYxjyXLZNLxZ420wbfjNLPf5bgXwlJllERAlPBadtOrD8HTKXU3e1hTeWwRnjsfv4Hh0xlWLxpo4mKpJ18SKVyVBF0PCII5o2LO2yG74zTT35KukcsCqzReKQJxwDHNKtF0RSNsVE2KJPVVLZJ7F70NqJYAFVVkEyulklHj8Jl5ZFsFXN8EcrR62G04xjAYxjAYxjAYxjAYxjAYxjAYxjAZxL6fNLHLrey45ZBFG5ZXkPRFLxgtzxwPfj3ztucG/ul/33Q/yJf95MDTj3R0kgBGpTTFVuRJJ4dQQS09UyFAxqOPhQf30e3MgdwtGdo++kSchSXEZBJnkj3UkxKqEVWO78tbIvy8+xgb5pe5OjnMSwa8bpI9+1ljLAKsTPdSja9SMdnNCJ7bi8vfohU6TX6+MShfDIhMjqF8o1KoWKsaUkDoSavOUwzMjBkYqw5BUkEH5EdM65/c4+bW6vdzcFm+b/CLgdc3iMtqYdSNRI7qhjUqRLxwiheFkAtt3Sgd3l5Wx7H06Oj71DVPPVi+srg/0ZLg7MiWV5gvncAX6AAVSjoL9fU0PYVX6HXpDEzPfm1MygAWSfFkY0B1pVY1144s0MCzOjiA+BQAD6DjncftsA/WMqkn0cXhhUHlRTuEbEqm2RlJO2/xH+Y5ySe3YqNXfNAgiwF3X68VX5xkSPtLR7OIhtC7gPDFUFEnA9KWUN/+w+u4AJMc+jZwoCFg23hLpkqhdUKJFc9T75YNo4yACi0DY4HBAq/zcZWntPTJvIjraeoQDcxYIK+ZbizXw30o5m+/cZvasjVfRfbxPQkHrG61V2OaBvA599NcYUaFVAAAlAA4AA8EAD5Yx9NbhhomBsESkEG7B8H1HGMDqmMYwGMYwMDaRCxcqCzJsPsygkgEdDRJ6/lH3OZPCW7oX71zx/5P58YwPngJ+SvTb0Hw+31fLPZUdaxjAxjToKpV8oocDgcGh7DgfmGF06DoiiuOg6Xf68YwPXhLYNCwKBrkD2Hy4GYk0MYdXCAMqlFrgKrFWYAdBZVSePQYxgSMYxgMYxgMYxgMYxgMYxgMYxgMYxgM479O2ijkm0m9bqN65I6svscYwOXfeWD8j/ab+3H3lg/I/wBpv7cYwH3lg/I/2m/tzo30HaKOPWzbFq9Ob5J6SR+5+eMYHbc5vq+2JkknjVhsE8hClEbnxGP4yn1xjAx/f/Uflr0r97j6cfxfkPzDPg7dn58y8gA/g4+QBQHw9KJFYxgfW7f1BsFlN9bjj59fyc+N27ORRZSP5uP5/wAX+M3/APR98YwNa78doSTiAStu2b9vAWr2fkgewxjGB//Z)

#### Pros
- Is able to recognise class probability for agricultural lands, industrial, public, commercial land and airports, sports and leisure facilities quite well. 
- Is able to recognise features and similarities between different cities, pointing to quantitive similarities of classes between cities.

#### Cons
- Not that great at roads (since roads have different functional classification for urban planning purposes. 
- Labels used were high-level, more abstract (low density urban fabric and sports and leisure facilities)
- Better results when architecture was trained over a mixture of cities then if trained on one city and tested on another.

