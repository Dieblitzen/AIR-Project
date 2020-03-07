## Credit: Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras import layers, models


"""
ResNet Basic Block, ususally used for smaller ResNets.
Uses 2 convolutions, and batch norm and ReLU.
Requires:
  out_channels: Number of output channels from the block
  stride: Integer or tuple (horizontal stride, vertical stride)
  downsample: Layer that will downsample input resolution /2
  base_width: Number of channels (depth) maintained throughout block.
  dilation: Integer or tuple for dilated convolutions
"""
class BasicBlock(Model):
  expansion = 1
  def __init__(self, out_channels, stride=1, downsample=None, base_width=64, dilation=1):
    super(BasicBlock, self).__init__()
    if base_width != 64:
      raise ValueError('BasicBlock only supports base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = layers.Conv2D(out_channels, (3, 3), strides=stride, padding='same')
    self.bn1 = layers.BatchNormalization(name='bn1')
    self.relu = layers.ReLU(name='relu')
    self.conv2 = layers.Conv2D(out_channels, (3, 3), strides=1, padding='same')
    self.bn2 = layers.BatchNormalization(name='bn2')
    self.downsample = downsample
    self.stride = stride

  def call(self, x, training=False):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out, training=training)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


"""
ResNet Bottleneck block, used for larger ResNets. 
Convolves the input 3 times, also using batch norm and ReLU
Requires:
  out_channels: Number of output channels from the block
  stride: Integer or tuple (horizontal stride, vertical stride)
  downsample: Layer that will downsample input resolution /2
  base_width: Number of channels (depth) used at start of block.
  dilation: Integer or tuple for dilated convolutions
"""
class BottleneckBlock(Model):
  # Expansion controls the #out_channels. expansion = 4 means for in_depth = 512, out_depth = 2048
  expansion = 4
  __constants__ = ['downsample']

  def __init__(self, out_channels, stride=1, downsample=None, base_width=64, dilation=1):
    super(BottleneckBlock, self).__init__()
    width = int(out_channels * (base_width / 64.))
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = layers.Conv2D(width, (1,1), padding='same') 
    self.bn1 = layers.BatchNormalization(name='bn1')
    self.conv2 = layers.Conv2D(width, (3,3), strides=stride, dilation_rate=dilation, padding='same')
    self.bn2 = layers.BatchNormalization(name='bn2')
    self.conv3 = layers.Conv2D(out_channels * self.expansion, (1, 1), padding='same')
    self.bn3 = layers.BatchNormalization(name='bn3')
    self.relu = layers.ReLU()
    self.downsample = downsample
    self.stride = stride

  def call(self, x, training=False):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out, training=training)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out
  

"""
Builds the layers of the ResNet model, without the fully connected final layer. 
Final output is 32x downsampled feature map of initial image.
- Can be initialised to Resnet50, Resnet101 etc.
- Can use atrous (dilated) convolutions instead of strides (NOT YET TESTED)
Requires:
  block: One of BasicBlock or BottleneckBlock (defines components of Resnet)
  layer_sizes: List of 4 numbers containing number of blocks to use per layer.
               eg: [3, 4, 6, 3] will use 3 blocks for layer1, 4 blocks for layer2 etc.
  width_per_group: #channels used within each block. Usually 64.
  replace_stride_with_dilation: list of booleans denoting which layer (after layer1)
                                to use dilated convolutions. Eg: [True, False, False]
"""
class ResNet():
  def __init__(self, block, layer_sizes, width_per_group=64, replace_stride_with_dilation=None):
    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                       "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.base_width = width_per_group

    conv1 = layers.Conv2D(self.inplanes, (7,7), strides=(2,2), padding='same', use_bias=False)
    bn1 = layers.BatchNormalization()
    relu = layers.ReLU()
    maxpool = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')

    self.layer0 = Sequential(layers=[conv1, bn1, relu, maxpool], name='layer0')
    self.layer1 = self._make_layer('layer1', block,  64, layer_sizes[0])
    self.layer2 = self._make_layer('layer2', block, 128, layer_sizes[1], stride=2,
                                    dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer('layer3', block, 256, layer_sizes[2], stride=2,
                                    dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer('layer4', block, 512, layer_sizes[3], stride=2,
                                    dilate=replace_stride_with_dilation[2])
    # PyTorch does some fancy initialising of weights for each layer, but we can avoid that for now.

  def _make_layer(self, name, block, out_channels, num_blocks, stride=1, dilate=False):
    ## Constructs a layer of blocks
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != out_channels * block.expansion:
      downsample = Sequential([
        layers.Conv2D(out_channels*block.expansion, (1,1), stride),
        layers.BatchNormalization()
      ])

    self.inplanes = out_channels * block.expansion

    blocks = [block(out_channels=out_channels, stride=stride, downsample=downsample, 
                    base_width=self.base_width, dilation=previous_dilation)]
    for _ in range(1, num_blocks):
      blocks.append(block(out_channels=out_channels, base_width=self.base_width, dilation=self.dilation))

    return Sequential(layers=blocks, name=name)


"""
Instantiates fresh ResNet model using functional API, so that intermediate layers 
can be queried.
Requires:
  block: One of BasicBlock or BottleneckBlock (defines components of Resnet)
  layer_sizes: List of 4 numbers containing number of blocks to use per layer.
               eg: [3, 4, 6, 3] will use 3 blocks for layer1, 4 blocks for layer2 etc.
  input_shape: Tuple/list denoting size of image (h, w, #channels)
               eg: (None, None, 3) for variable h, w and 3 rgb
  width_per_group: #channels used within each block. Usually 64.
  replace_stride_with_dilation: list of booleans denoting which layer (after layer1)
                                to use dilated convolutions. Eg: [True, False, False]
"""
def create_resnet(block, layer_sizes, input_shape=(None, None, 3), 
                  width_per_group=64, replace_stride_with_dilation=None):
  img_input = layers.Input(shape=input_shape, name='input')
  resnet = ResNet(block, layer_sizes, width_per_group, replace_stride_with_dilation)
  
  x = resnet.layer0(img_input)
  x = resnet.layer1(x)
  x = resnet.layer2(x)
  x = resnet.layer3(x)
  x = resnet.layer4(x)

  return Model(inputs=img_input, outputs=x)


def resnet18(**kwargs):
  """
  Contructs a fresh ResNet-18 model, using BasicBlocks.
  """
  resnet = create_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
  return resnet


def resnet34(**kwargs):
  """
  Contructs a fresh ResNet-34 model, using BasicBlocks.
  """
  resnet = create_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
  return resnet


def resnet50(**kwargs):
  """
  Contructs a fresh ResNet-50 model, using BottleneckBlocks.
  """
  resnet = create_resnet(BottleneckBlock, [3, 4, 6, 3], **kwargs)
  return resnet


def resnet101(**kwargs):
  """
  Contructs a fresh ResNet-101 model, using BottleneckBlocks.
  """
  resnet = create_resnet(BottleneckBlock, [3, 4, 23, 3], **kwargs)
  return resnet


def resnet152(**kwargs):
  """
  Contructs a fresh ResNet-152 model, using BottleneckBlocks.
  """
  resnet = create_resnet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
  return resnet