import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers, models


"""
Residual Convolution Unit
- Essentially a resnet block without the batch norm.
- Uses 3x3 convolutions (maintains dimensions of input.)
Requires:
  out_channels: number of channels (depth) of output tensor.
  n_layers: the number of convolutional layers within the the block
  kernel_size: (kernel_height, kernel_width)
  strides: (horizontal_stride, vertical_stride)
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
"""
class RCU_Block(Model):
  def __init__(self, out_channels, n_layers=2, 
               kernel_size=(3,3), strides=(1,1), padding='same'):
    super(RCU_Block, self).__init__()

    # Define as sequential list of layers
    self.rcu_block = Sequential()
    for _ in range(n_layers):
      self.rcu_block.add(layers.ReLU())
      self.rcu_block.add(layers.Conv2D(out_channels, kernel_size, strides=strides, padding=padding))
  
  def call(self, input_t):
    identity = input_t
    x = self.rcu_block(input_t)
    return x + identity


"""
Multi-resolution Fusion.
- Fuses inputs into high-res feature map. First applies (3x3) convolutions to create feature maps
  of same depth dimension (smallest depth of channels among inputs).
- Upsamples the smaller feature maps to largest resolution of inputs, then sums them all.
Requires:
  out_channels: number of channels (depth) of output tensor.
  num_inputs: the number of input tensors to the MRF block
  use_deconv: Whether to upsample using (learnable) conv_transpose layers, or only bilinear.
  kernel_size: (kernel_height, kernel_width) for both conv and conv_transpose operations.
  strides: (horizontal_stride, vertical_stride)
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
"""
class MRF_Block(Model):
  def __init__(self, out_channels, num_inputs, use_deconv=True, 
               kernel_size=(3,3), strides=(1,1), padding='same'):
    super(MRF_Block, self).__init__()

    # Add the conv/deconv layers.
    for i in range(num_inputs):
      # Convolve input tensors to output tensors of same channel depth (# of channels)
      conv_layer = layers.Conv2D(out_channels, kernel_size, strides=strides, padding=padding)
      setattr(self, f'conv_{i}', conv_layer)

      # Deconv input tensors. Input tensor at index i will be upsampled by factor 2^(n_inputs - (i+1))
      # Eg: if input tensors are [t1, t2, t3], t1 will be upsampled x 4, t2 upsampled x 2, t3 not upsampled.
      if use_deconv:
        deconv_layer = layers.Conv2DTranspose(out_channels, kernel_size, padding='same',
                                              strides=(2**(num_inputs-(i+1)), 2**(num_inputs-(i+1))))
        setattr(self, f'deconv_{i}', deconv_layer)

  ## inputs: a list of tensors being inputted to the block.
  def call(self, inputs):

    # Make #channels the same for input tensors (smallest depth)
    convolved = []
    for i, t in enumerate(inputs):
      conv = getattr(self, f'conv_{i}')
      convolved.append(conv(t))

    # Upsample to largest (h, w) resolution
    largest_res = max(inputs, key=lambda t: t.shape[1] * t.shape[2])
    resized = []
    for i, t in enumerate(convolved):
      deconv = getattr(self, f'deconv_{i}', lambda t: t)
      up_sampled = deconv(t)
      
      # Use bilinear resizing if resolution of optional conv_transpose doesn't match
      if up_sampled.shape[1:3] != largest_res.shape[1:3]:
        up_sampled = tf.image.resize(t, size=largest_res.shape[1:3])
    
      resized.append(up_sampled)

    # Fuse by summing
    return sum(resized)


"""
Chained Residual Pooling.
- Chain of multiple pooling blocks, each consisting of one max-pooling layer
  and one convolutional layer. Kernel sizes: for pooling is 5, convolution 3.
- Output feature maps of pooling blocks are summed with identity mappings.
- Maintains the dimensions of the input.
Requires:
  out_channels: number of channels (depth) of output tensor (should be same as input to block)
  n_pool_blocks: number of CRP blocks to apply
  k_size_pool: (kernel_height, kernel_width) for pooling operation. Usually (5,5).
  k_size_conv: (kernel_height, kernel_width) for conv operation. Usually (3,3).
  strides: (horizontal_stride, vertical_stride) for convolution layer.
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
"""
class CRP_Block(Model):
  def __init__(self, out_channels, n_pool_blocks=2, 
              k_size_pool=(5,5), k_size_conv=(3,3), strides=(1,1), padding='same'):
    super(CRP_Block, self).__init__()

    self.relu = layers.ReLU()

    # Create pool blocks. Padding for pooling operation is 'same'
    self.n_pool_blocks = n_pool_blocks
    for i in range(n_pool_blocks):
      pool = layers.MaxPool2D(pool_size=k_size_pool, strides=(1,1), padding='same')
      conv = layers.Conv2D(out_channels, k_size_conv, strides=strides, padding=padding)
      setattr(self, f'pool_block_{i}', Sequential(layers=[pool, conv]))


  ## input_t is the output of the mrf_block. 
  def call(self, input_t):
    result = self.relu(input_t)

    x = input_t
    for i in range(self.n_pool_blocks):
      pool_block = getattr(self, f'pool_block_{i}')
      x = pool_block(x)
      result += x
    
    return result


"""
RefineNet block.
- Applies Residual Convolution Units twice to each input tensor
- Fuses them together with Multi-Resolution Fusion.
- Applies Chained Residual Pooling
- Applies Residual Convolution Unit once one last time.
Requires:
  channels: list of output channels [c1, c2, ...]. Each c_i is output #channels (depth) of 
    layer i that is fed into this RefineNet Block.
  rcu/mrf/crp_kwargs: optional dictionary of keyword arguments for each of RCU, MRF, CRP blocks.
"""
class RefineNet_Block(Model):
  def __init__(self, channels, rcu_kwargs={}, mrf_kwargs={}, crp_kwargs={}):
    super(RefineNet_Block, self).__init__()

    for i, c in enumerate(channels):
      rcu_block = RCU_Block(out_channels=c, **rcu_kwargs)
      setattr(self, f'rcu_block_{i}', rcu_block)
    
    self.out_channels = min(channels)

    self.mrf = MRF_Block(out_channels=self.out_channels, num_inputs=len(channels), **mrf_kwargs)

    self.crp = CRP_Block(out_channels=self.out_channels, **crp_kwargs)

    self.rcu_final = RCU_Block(out_channels=self.out_channels, n_layers=2)
  
  def call(self, inputs):
    rcu_out = []
    for i, t in enumerate(inputs):
      rcu_block = getattr(self, f'rcu_block_{i}')
      rcu_out.append(rcu_block(t))
    
    mrf_out = self.mrf(rcu_out)

    crp_out = self.crp(mrf_out)

    out = self.rcu_final(crp_out)

    return out


"""
RefineNet Image Segmentation Model.
- Takes in a (pre-trained or not) backbone model (assumes channels_last data format).
- Upsamples from different resolutions of the backbone using RefineNet_Blocks.
- Returns a segmented image (h, w, #c), where h,w are input image resolution, #c = #classes.
Requires:
  backbone: A (pre-trained or not) backbone model (eg: ResNet)
  refine_net_blocks: [[layer4_name, layer3_name], [layer2_name], ...]
    Length of outer lists is how many RefineNet blocks to use.
    Inner lists denote backbone layer names that are relevant to RefineNet upsampling.
    All inner lists (except first) use previous RefineNet's output as an input as well.
  input_shape: Tuple/list denoting size of image (h, w, #channels)
  num_classes: The number of classes #c. This denotes the output size.
  rcu/mrf/crp_kwargs: optional dictionary of keyword arguments for each of RCU, MRF, CRP blocks.
"""
def create_refine_net(backbone, refine_net_blocks, num_classes, input_shape=(None, None, 3),
                      rcu_kwargs={}, mrf_kwargs={}, crp_kwargs={}):
  # Define the downsampling using the backbone model.
  intermediate_layers = [layer_name for block in refine_net_blocks for layer_name in block]
  intermediate_out = {name: backbone.get_layer(name).output for name in intermediate_layers}
  feature_extract = Model(inputs=backbone.input, outputs=intermediate_out)

  # Extract intermediate features by downsampling
  img_input = layers.Input(shape=input_shape, name='input')
  features = feature_extract(img_input)

  # Construct RefineNet on intermediate feature output and previous RefineNet output
  prev_refine_net_out = None
  for layer_names in refine_net_blocks:
    input_features = [features[name] for name in layer_names]
    input_channels = [feature.shape[-1] for feature in input_features]

    if prev_refine_net_out is not None:
      input_features.append(prev_refine_net_out)
      input_channels.append(prev_refine_net_out.shape[-1])
    
    refine_net_block = RefineNet_Block(input_channels, rcu_kwargs, mrf_kwargs, crp_kwargs)
    refine_net_out = refine_net_block(input_features)
    prev_refine_net_out = refine_net_out
  
  # Reduce number of channels in final convolution, and then resize to original resolution.
  x = tf.image.resize(prev_refine_net_out, size=(input_shape[0], input_shape[1]))
  x = layers.Conv2D(num_classes, (1,1), strides=(1,1), padding='same')(x)
  
  return Model(inputs=img_input, outputs=x)
