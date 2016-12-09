<a name="nn.modules"/>
# DIANNE module catalog #
  * [I/O](#nn.io)
    * [Input](#nn.Input)
    * [Output](#nn.Output)
  * [Layer](#nn.layer)
    * [Convolution](#nn.Convolution) (trainable)
    * [FullConvolution](#nn.FullConvolution) (trainable)
    * [Linear](#nn.Linear) (trainable)
    * [MaxPooling](#nn.MaxPooling)
    * [MaskedMaxPooling](#nn.MaskedMaxPooling)
    * [MaxUnpooling](#nn.MaxUnpooling)
    * [AvgPooling](#nn.AvgPooling)
    * [Reshape](#nn.Reshape)
    * [Zeropad](#nn.Zeropad)
  * [Activation](#nn.activation)
    * [ReLu](#nn.ReLu)
    * [RReLu](#nn.PReLu) (trainable)
    * [Sigmoid](#nn.Sigmoid)
    * [Softmax](#nn.Softmax)
    * [LogSoftmax](#nn.LogSoftmax)
    * [Tanh](#nn.Tanh)
    * [Threshold](#nn.Threshold)
  * [Fork](#nn.fork)
    * [Duplicate](#nn.Duplicate)
    * [Grid](#nn.Grid)
    * [Split](#nn.Split)
  * [Join](#nn.join)
    * [Accumulate](#nn.Accumulate)
    * [Average](#nn.Average)
    * [Concat](#nn.Concat)
    * [Multiply](#nn.Multiply)
  * [Preprocessing](#nn.preprocessing)
    * [Normalization](#nn.Normalization)
    * [Denormalization](#nn.Denormalization)
    * [Frame](#nn.Frame)
    * [Narrow](#nn.Narrow)
    * [Scale](#nn.Scale)
  * [Regularization](#nn.regularization)
    * [BatchNormalization](#nn.BatchNormalization) (trainable)
	* [Dropout](#nn.Dropout)
	* [DropPath](#nn.DropPath)
  * [Composite](#nn.composite)
    * [Inception_v1](#nn.Inception_v1) (trainable)
    * [Inception_v2](#nn.Inception_v2) (trainable)
    * [Residual](#nn.Residual) (trainable)
    * [Residual_b](#nn.Residual_b) (trainable)
    * [Fractal-3](#nn.Fractal-3) (trainable)
  * [References](#nn.references)

<a name="nn.io"/>
## I/O ##
I/O-modules form the entry- and exitpoints of the model/computational graph.

<a name="nn.Input"/>
### Input ###
Entrypoint of the computational graph. Inputsamples are forwarded into the model using these modules. Gradient calculations also terminate here. Note that a single model can in general have multiple input modules.

<a name="nn.Output"/>
### Output ###
Exitpoint of the computational graph. Here, the output of the network is reported and gradients can be injected for backpropagation. Note that a single model can in general have multiple output modules.

<a name="nn.layer"/>
## Layer ##
These are general layers transforming a single input tensor to a single output tensor, usually changing the dimensions.

<a name="nn.Convolution"/>
### Convolution ###
Performs a convolution on a number of input planes (feature maps, channels, ...) and generates a number of output planes. Only the "valid" region is output, so output planes will be smaller than input planes for kernel sizes > 1. A scalar bias is added to each output plane as well. Depending on the which parameters are set, this module performs a temporal, spatial or volumetric convolution:

  * `noInputPlanes` : number of input planes
  * `noOutputPlanes` : number of output planes
  * `kernelWidth` : kernel size in the first dimension
  * `kernelHeight` : kernel size in the second dimension (only for spatial & volumetric)
  * `kernelDepth` : kernel size in the third dimension (only for volumetric)
  * `strideX` : stride in the first dimension
  * `strideY` : stride in the second dimension (only for spatial & volumetric)
  * `strizeZ` : stride in the third dimension (only for volumetric)
  * `padX` : zero-padding in the first dimension
  * `padY` : zero-padding in the second dimension (only for spatial & volumetric)
  * `padZ` : zero-padding in the third dimension (only for volumetric)

In the spatial case, this module takes a 3- or 4-dimensional tensor as input and generates a 3- or 4-dimensional tensor as output. Indices are assumed to be ([batch,] plane, x, y).
This is a trainable module.

<a name="nn.FullConvolution"/>
### FullConvolution ###
Performs a full convolution on a number of input planes (feature maps, channels, ...) and generates a number of output planes. The full region is output, so output planes will be larger than input planes for kernel sizes > 1. A scalar bias is added to each output plane as well. Depending on the which parameters are set, this module performs a spatial or volumetric convolution (temporal is currently not supported):

  * `noInputPlanes` : number of input planes
  * `noOutputPlanes` : number of output planes
  * `kernelWidth` : kernel size in the first dimension
  * `kernelHeight` : kernel size in the second dimension
  * `kernelDepth` : kernel size in the third dimension (only for volumetric)
  * `strideX` : stride in the first dimension
  * `strideY` : stride in the second dimension
  * `strizeZ` : stride in the third dimension (only for volumetric)
  * `padX` : zero-padding in the first dimension
  * `padY` : zero-padding in the second dimension
  * `padZ` : zero-padding in the third dimension (only for volumetric)

In the spatial case, this module takes a 3D or 4D tensor as input and generates a 3D or 4D tensor as output. Indices are assumed to be ([batch,] plane, x, y).
This is a trainable module.

<a name="nn.Linear"/>
### Linear ###
Calculates a tensor of linear combinations of the input tensor and adds a bias tensor to the output. Parameters are:

  * `input` : number of input elements
  * `outut` : number of output elements

The dimensions of the input don't matter, as long as the number of elements matches. In case of batches, the first dimension is assumed to be the batch dimension. The output will be a 1D tensor (or 2D in case of batches).
This is a trainable module.

<a name="nn.MaxPooling"/>
### MaxPooling ###
Performs a plane-wise pooling operation by outputting the maximum over a sliding window of the input planes. The output planes will hence generally be smaller than the input planes. Optionally, the indices of the selected elements can be passed as a second output. Depending on which parameters are set, the maxpooling can be temporal, spatial or volumetric:

  * `width` : window size in the first dimension
  * `height` : window size in the second dimension (only for spatial & volumetric)
  * `depth` : window size in the third dimension (only for volumetric)
  * `strideX` : stride in the first dimension
  * `strideY` : stride in the second dimension (only for spatial & volumetric)
  * `strizeZ` : stride in the third dimension (only for volumetric)

In the spatial case, this module takes a 3D or 4D tensor as input and generates a 3D or 4D tensor as output. Indices are assumed to be ([batch,] plane, x, y).

<a name="nn.MaskedMaxPooling"/>
### MaskedMaxPooling ###
Performs a maxpooling operation according to a given mask. Parameters are:

  * `noInputs` : number of elements in the input tensor
  * `masks` : a list of indices over which the maximum should be pooled

For example, `0,2,4;1,3,5` will take an input tensor with (at least) 6 elements and generate a 1D output tensor of size 2 with the maximum of elements on the even and uneven indices.
Note: this module does not check if the input is a batch or not!

<a name="nn.MaxUnpooling"/>
### MaxUnpooling ###
Performs the reverse operation of the [MaxPooling](#nn.MaxPooling) module and requires two input tensor: en value tensor and an index tensor (as generated by [MaxPooling](#nn.MaxPooling)). The output tensor is generated by putting the values on the corresponding indices in a zero-initialised tensor. The output planes will hence generally be larger than the input planes. Depending on which parameters are set, the maxunpooling can be temporal, spatial or volumetric:

  * `width` : window size in the first dimension
  * `height` : window size in the second dimension (only for spatial & volumetric)
  * `depth` : window size in the third dimension (only for volumetric)
  * `strideX` : stride in the first dimension
  * `strideY` : stride in the second dimension (only for spatial & volumetric)
  * `strizeZ` : stride in the third dimension (only for volumetric)

In the spatial case, this module takes a 3D or 4D tensor as input and generates a 3D or 4D tensor as output. Indices are assumed to be ([batch,] plane, x, y).

<a name="nn.AvgPooling"/>
### AvgPooling ###
Performs a plane-wise pooling operation by outputting the average over a sliding window of the input planes. The output planes will hence generally be smaller than the input planes. Depending on which parameters are set, the average pooling can be temporal, spatial or volumetric:

  * `width` : window size in the first dimension
  * `height` : window size in the second dimension (only for spatial & volumetric)
  * `depth` : window size in the third dimension (only for volumetric)
  * `strideX` : stride in the first dimension
  * `strideY` : stride in the second dimension (only for spatial & volumetric)
  * `strizeZ` : stride in the third dimension (only for volumetric)

In the spatial case, this module takes a 3D or 4D tensor as input and generates a 3D or 4D tensor as output. Indices are assumed to be ([batch,] plane, x, y).

<a name="nn.Reshape"/>
### Reshape ###
Reshapes an input tensor to an arbitrarily dimensioned output tensor with the same number of elements. Dimensions are set as follows:

  * `dim0` : size of first dimension
  * `dim1` : size of second dimension (optional)
  * ...

In the case the number of elements iof the input s a multiple of the expected number of elements, batch operation is assumed with the first dimension being treated as the batch dimension (so the output will be of dimension batch x dim0 x dim1 x ...).

<a name="nn.Zeropad"/>
### Zeropad ###
Pads the input tensor with zeros in the last dimensions. According to which parameters are set, this can either be spatial or volumetric padding:

  * `dim0` : padding in the last dimension
  * `dim1` : padding in the second to last dimension
  * `dim2` : padding in the third to last dimension (only for volumetric)

In the spatial case, this module takes a 3D or 4D tensor as input and generates a 3D or 4D tensor with dimensions [batch x] plane x (height+2\*dim1) x (width+2\*dim0).

<a name="nn.activation"/>
## Activation ##
These modules perform element-wise operations on the input tensor, and hence the output has the same dimensions as the input.

<a name="nn.ReLu"/>
### ReLu ###
Calculates the Rectified Linear Unit [1] activation function, i.e. x if x > 0, 0 otherwise. This module has no parameters.

<a name="nn.PReLu"/>
### PReLu ###
Calculates the Parametric Rectified Linear Unit [2] activation function, i.e. x if x > 0, a*x otherwise. This module has no parameters, but is trainable.

<a name="nn.Sigmoid"/>
### Sigmoid ###
Calculates the sigmoid or logistic activation function, i.e. 1/(1 + exp(-x)). This module has no parameters.

<a name="nn.Softmax"/>
### Softmax ###
Calculates the softmax activation function, which squashes values between 0 and 1, with the sum always equal to 1. This effectively creates a probability distribution. This module performs differently depending on the dimension of the input tensor:

  * 1D: softmax over all inputs
  * 2D: softmax over each row (e.g. in case of batch operation)
  * 3D: pixelwise softmax over first dimension
  * 4D: pixelwise softmax over second dimension, for each index in first dimension

This module has no parameters.

<a name="nn.LogSoftmax"/>
### LogSoftmax ###
Same as [Softmax](#nn.Softmax), except values are log-probabilities and it only accepts 1D or 2D tensors. This module has no parameters.

<a name="nn.Tanh"/>
### Tanh ###
Calculates the hyperbolic tangent activation function. This module has no parameters.

<a name="nn.Threshold"/>
### Threshold ###
Thresholds the input tensor according to:

  * `thresh` : the value to threshold at
  * `val` : the value to replace the input with

The output is then x if x > `thresh`, `val` otherwise.

<a name="nn.fork"/>
## Fork ##
Fork modules take a single output but generate multiple output tensors, which are independently forwarded to (possibly different) succeeding modules, allowing branching in the computational graph.

<a name="nn.Duplicate"/>
### Duplicate ###
Creates an exact duplicate of the input and forwards it for each of the succeeding modules. This module has no parameters.

<a name="nn.Grid"/>
### Grid ###
Performs a sliding window crop on the last two dimensions of the input tensor and forwards each crop sequentially to the (single) succeeding module. This module has following parameters:

  * `x` : size of the crop in the second to last dimension
  * `y` : size of the crop in the last dimension
  * `strideX` : stride of the crop in the second to last dimension
  * `strideY` : stride of the crop in the last dimension

Not that the backwards operation is not supported by this module.

<a name="nn.Split"/>
### Split ###
Splits the input tensor evenly by the number of succeeding modules along a certain dimension and forwards each "slice" to a distinct module. This module has following parameters:

  * `dim` : the dimension along which to split, counting backwards

For example, if you want to split a 4D tensor along the third dimension, you need to set `dim` to 2. This allows for transparant batch operation.
Note that the order in which the succeeding modules are defined is important here.

<a name="nn.join"/>
## Join ##
Join modules combine multiple input tensor from possibly distinct preceding modules into a single output tensor, joining multiple branches of the computational graph.

<a name="nn.Accumulate"/>
### Accumulate ###
Accumulates (elementwise addition) the input tensors into the output tensor. This module has no parameters.

<a name="nn.Average"/>
### Average ###
Averages the input tensors into the output tensor. This module has no parameters.

<a name="nn.Concat"/>
### Concat ###
Concatenates the input tensor along a certain dimension into the output tensors. This module has following parameters:

  * `dim` : the dimension along which to concatenate, counting backwards

For example, if you want to concatenate 4D tensors along the third dimension, you need to set `dim` to 1. This allows for transparant batch operation.

<a name="nn.Multiply"/>
### Multiply ###
Performs elementwise multiplication of the input tensors into the output tensor. This module has no parameters.

<a name="nn.preprocessing"/>
## Preprocessing ##
Preprocessing modules generally perform some operations on the input or output tensors making them suitable for processing by or improving training of the neural network. While peprocessing modules are not trainable in the strict sense, they do possibly have parameters/weights that need to be initialised before the optimisation process. Therefore, optimisation routines first have to call the `isPreprocessed()` and `preprocess(Dataset data)` methods in order to set these parameters.

<a name="nn.Normalization"/>
### Normalization ###
This module will offset and rescale the input tensors to have zero mean and unit variance (standard normal distribution). Note that at the moment a single mean and variance are calculated for all position of the input tensor. This module needs to be initialised before optimisation.

<a name="nn.Denormalization"/>
### Denormalization ###
This module will rescale and offset the output of the neural network with the variance and mean of the dataset target tensors, so the network itself can learn a more stable zero mean and unit variance output. This module needs to be initialised before optimisation.

<a name="nn.Frame"/>
### Frame ###
This module will take a tensor and crop and rescale it to fit the given dimensions. Currently this module only supports 3D framing:

  * `dim0` : desired size in first dimension
  * `dim1` : desired size in second dimension
  * `dim2` : desired size in third dimension (optional)

For example, if a neural network expects an RGB image input of dimensions 231x231, set `dim0` tot 3 and both `dim1` and `dim0` to 231. Any RGB image will than be transformed into a valid input. A 4D tensor is accepted as well, with the first dimension used as batch.

<a name="nn.Narrow"/>
### Narrow ###
Slices the input tensor to fit the given dimensions. Currently this module only supports 3D slicing:

  * `index0` : start position in the first dimension
  * `index1` : start position in the second dimension
  * `index2` : start position in the third dimension (optional)
  * `size0` : desired size in the first dimension
  * `size1` : desired size in the second dimension
  * `size2` : desired size in the third dimension (optional)

A 4D tensor is accepted as well, with the first dimension used as batch.

<a name="nn.Scale"/>
### Scale ###
Rescales the entire input tensor to fit the given dimensions using bilinear interpolation. Currently this module only supports 3D scaling:

  * `dim0` : desired size of the first dimension
  * `dim1` : desired size of the second dimension
  * `dim2` : desired size of the third dimension

A 4D tensor is accepted as well, with the first dimension used as batch.

<a name="nn.regularization"/>
## Regularization ##
These modules do some form of regularisation in order to improve the training of the model.

<a name="nn.BatchNormalization"/>
### BatchNormalization ###
Performs batch normalisation [3] on the input (output of preceding module). This module has a single parameter:

  * `size` : the number of features or feature maps in the input

This module currently supports singular features and spatial or volumetric feature maps and will behave differently according to the the input dimension:

  * 1D: single 1D input with `size` singular features
  * 2D: batch of 2D inputs with `size` singular features
  * 3D: single 3D input with `size` spatial feature maps
  * 4D:
    * If the second dimension equals `size`, batch of 3D inputs with `size` spatial feature maps
    * Otherwise, single 4D input with `size` volumetric feature maps
  * 5D: batch of 4D inputs with `size` volumetric feature maps

This is a trainable module.

<a name="nn.Dropout"/>
### Dropout ###
Performs elementwise dropout [4] on the input (output of the preceding module). This module has a single parameter:

  * `rate` : the dropout rate, i.e. probability of an value being set to zero (in [0,1])

<a name="nn.DropPath"/>
### DropPath ###
Performs tensorwise droppath [7] on the input tensor, i.e. with a certain probability it will set the full input to zero or will pass it unchanged otherwise. This module has a single parameter:

  * `rate` : the droppath rate, a real value in [0,1]

<a name="nn.composite"/>
## Composite ##
Composite modules are combinations of other modules in this catalog, and are in turn building blocks for larger models. When deployed, composite modules are "expanded" into their constituting modules, which can then in turn be individually deployed. You can create your own composite modules by adding a `composite.txt` file to a model folder, specifying the configuration options (e.g. number of filters). Have a look at the [`be.iminds.iot.dianne.nn.module`](../be.iminds.iot.dianne.nn.module/composites) project for examples.

<a name="nn.Inception_v1"/>
### Inception_v1 ###
Inception layer as described in [5]. This module only works on (batched) spatial inputs, so 3D or 4D tensors. Following parameters need to be set:

  * `noInputPlanes` : number of input planes (feature maps)
  * `1x1` : number of feature maps created by 1x1 convolutions
  * `3x3` : number of feature maps created by 3x3 convolutions
  * `5x5` : number of feature maps created by 5x5 convolutions
  * `3x3reduce` : number of feature maps generated by 1x1 convolutions preceding 3x3 convolutions
  * `5x5reduce` : number of feature maps generated by 1x1 convolutions preceding 5x5 convolutions
  * `poolproj` : number of feature maps created by 1x1 convolutions succeeding maxpooling

This is a trainable module.

<a name="nn.Inception_v2"/>
### Inception_v2 ###
Same as [Inception_v1](#nn.Inception_v1), but including batch normalisation [3]. This is a trainable module.

<a name="nn.Residual"/>
### Residual ###
Residual layer as described in [6]. This module only works on (batched) spatial inputs, so 3D or 4D tensors, and includes batch normalisation [3]. Following parameters need to be set:

  * `noInputPlanes` : number of input planes (feature maps)
  * `noOutplanes` : number of output planes (feature maps)

This is a trainable module.

<a name="nn.Residual_b"/>
### Residual_b ###
Same as [Residual](#nn.Residual), but including an additional layer of 1x1 convolutions before 3x3 convolutions to reduce dimensionality:

  * `noFilters` : number of feature maps after 1x1 convolutions

This is a trainable module.

<a name="nn.Fractal-3"/>
### Fractal-3 ###
Fractal layer as described in [7], unfolded to C = 3. This module only works on (batched) spatial inputs, so 3D or 4D tensors, and includes batch normalisation [3]. Following parameters need to be set:

  * `noInputPlanes` : number of input planes (feature maps)
  * `noOutputPlanes` : number of feature maps after all 3x3 convolutions

This is a trainable module.

<a name="nn.references"/>
## References ##
[1] [Rectified Linear Units Improve Restricted Boltzmann Machines](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)

[2] [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

[3] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[4] [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html)

[5] [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

[6] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[7] [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/abs/1605.07648)
