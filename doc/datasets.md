<a name="dataset"/>
# Datasets and adapters #
  * [Automatically-configured datasets](#dataset.auto)
    * [FileDataset](#dataset.FileDataset)
    * [CSVDataset](#dataset.CSVDataset)
	* [ImageClassificationDataset](#dataset.ImageClassificationDataset)
	* [ImageSegmentationDataset](#dataset.ImageSegmentationDataset)
  * [Dataset adapters](#dataset.adapters)
    * [RangeAdapter](#dataset.RangeAdapter)
    * [RandomFlipAdapter](#dataset.RandomFlipAdapter)
    * [RandomRotationAdapter](#dataset.RandomRotationAdapter)
    * [RandomCropAdapter](#dataset.RandomCropAdapter)
    * [FrameAdapter](#dataset.FrameAdapter)
    * [AutoencoderAdapter](#dataset.AutoencoderAdapter)
    * [LabelAdapter](#dataset.LabelAdapter)
    * [MaskToClassAdapter](#dataset.MaskToClassAdapter)
    * [BinarizeAdapter](#dataset.BinarizeAdapter)

<a name="dataset.auto"/>
## Automatically-configured datasets ##
DIANNE will automatically track a folder (`tools/datasets` by default) for dataset directories and bring a corresponding `Dataset` service online to be used e.g. for training. A number of well-known datasets are supported out-of-the-box. They are listed on the DIANNE [datasets page](http://dianne.intec.ugent.be/datasets/) and can be downloaded using Gradle by executing:

	./gradlew dataset -Pwhich=<name>
	
However, you can also add your own dataset without the need of writing a custom `Dataset` service if your files adhere to certain rules. In order to do so, you need to create a directory for the dataset and add one or more JSON descriptors to it that tell DIANNE how it should interpret the data. For every descriptor, a service will be created. Each descriptor should contain at least the following items:

  * `name` : The name of the dataset.
  * `type` : One of the types listed below.
  * `inputDims` : Dimensions of the inputs as an integer array, e.g. [3,32,32] for CIFAR-10.
  * `noSamples` : Number of samples in the dataset. 

There are also a number of items that are only required for supervised learning:

  * `targetDims` : Dimensions of the targets as an integer array, e.g. [10] for CIFAR-10.
  * `labelsFile` : Labels for the different target dimensions, if applicable (as for example with classification). This is optional.

Currently we support the following automatic configurations.

<a name="dataset.FileDataset"/>
### FileDataset ###
Reads a number of files as raw blob of unsigned bytes for classification. Input data is automatically rescaled from [0,255] to [0,1]. Targets are interpreted as class indices (so a maximum of 256 labels is supported), and are converted to a one-hot encoding. Following options need to be set:

  * `inputFiles` : A string array with the input data files. Multiple samples can be in the same file.
  * `targetFiles` : A string array with the target data files. Each input file needs to have a matching target file. Each byte in the target file is read as a separate class index, labeling the corresponding input bytes in the matching input data file.
  * `files` : Can be used instead of `inputFiles` and `targetFiles` when the inputs and labels are in the same file. It is assumed that the class index comes first, and is followed by the input bytes.
  * `prefetch` : Whether to prefetch the data. This can speedup the start of the learning, but increases memory usage.

<a name="dataset.CSVDataset"/>
### CSVDataset ###
Reads a single ASCII file for supervised learning. Each line is interpreted as a separate sample, containing both the input and the target data as delimited floating point or integer numbers. Following options need to be set:

  * `file` : The ASCII file to be read.
  * `separator` : The character used to separate the different values. By default this is ",", hence the name.
  * `inputOffset` : The column at which to start reading the input, counting from the start of the line.
  * `targetOffset` : The column at which to start reading the target, counting from the end of the input (so use 0 if consecutive).
  * `classification` : Boolean value indicating if this dataset is used for classification. If `true`, a single integer target value is read and interpreted as the class index. This is then converted into a one-hot encoding.
  
**Note**: You can optionally omit the `noSamples` options, as this can be determined automatically by counting the number of lines in the file.

<a name="dataset.ImageClassificationDataset"/>
### ImageClassificationDataset ###
Reads a number of JPEG images for image classification. The input images need to be put in a subdirectory called `images`, and have a name `<index>.jpg`, with `<index>` being the index in the set. Following options need to be set:

  * `targetsFile` : File containing the class indices on separate lines. These indices are then converted to one-hot encoding.

<a name="dataset.ImageSegmentationDataset"/>
### ImageSegmentationDataset ###
Reads a number of JPEG images for image segmentations. The input images need to be put in a subdirectory called `images`, and have a name `<index>.jpg`, with `<index>` being the index in the set. Similarly, the masks need to be put in a subdirectory called `masks`, and have a name `<index>.jpg`, with `<index>` being the index in the set. No further options need to be set.

<a name="dataset.adapters"/>
## Dataset adapters ##
Dataset adapters are a mechanism in DIANNE for changing an existing dataset on-the-fly, for example for data augmentation. These adapters are true `Dataset` services and can thus be used wherever a regular dataset can be used, such as in learning jobs. There are two ways of activating these adapters:
* Either create a JSON descriptor file, which needs following entries (besides the ones required for the adapter itself):
  * `name` : The name of the adapter dataset.
  * `adapter` : One of the types of adapters listed below.
  * `dataset` : The name of the dataset that needs to be adapted. *Note*: this can itself be an adapter!
* Specify adapter options listed below marked with an (*) in the job options when submitting a job to the `DianneCoordinator`. This will automatically activate the required adapters, in the order as mentioned in this list.

<a name="dataset.RangeAdapter"/>
### RangeAdapter ###
Can be used to create a subset of the dataset samples. The can be used to e.g. separate training, validation and testing sets. Required options are:

  * `range`(*) : Integer array specifying the samples that are allowed. Can be specified as either `[max]` for indices in [0,max], `[min,max]` or by specifying only valid indices `[i,j,...]`. `max` and `min,max` (without brackets) are also possible.

<a name="dataset.RandomFlipAdapter"/>
### RandomFlipAdapter ###
Will perform a random horizontal and or vertical flip on the underlying image dataset. Inputs are assumed to be 2D or 3D tensors. Required options are:

  * `vlip`(*) : Probability of performing a vertical flip (default is 0).
  * `hflip`(*) : Probability of performing a horizontal flip (default is 0).

<a name="dataset.RandomRotationAdapter"/>
### RandomRotationAdapter ###
Performs random rotation on the underlying image dataset. Inputs are assumed to be 2D or 3D tensors. Required options are:

  * `rotationTheta`(*) : Either a single real value `theta` resulting in uniform random rotation in [-theta, theta] or an integer array `[min_theta,max_theta]` resulting in rotations in [-min_theta, max_theta]. Defaults to [-pi, pi].
  * `center` : Point of rotation specified as integer array `[x,y]`. By default, the point of rotation is randomly selected.
  * `middle` : Boolean value for using the center of the image as the point of rotation (default is `false`). This is overwritten by `center`.

<a name="dataset.RandomCropAdapter"/>
### RandomCropAdapter ###
Performs a random crop on the underlying image dataset. Inputs are assumed to be 2D or 3D tensors. Required options are:

  * `cropWidth`(*) : Either a single integer value resulting in crops with fixed width or an integer array `[min,max]` resulting in crops with uniform random width in [min,max].
  * `cropHeight`(*) : Either a single integer value resulting in crops with fixed height or an integer array `[min,max]` resulting in crops with uniform random height in [min,max].
  * `cropPadding` : Integer value for optional zero-padding on the sides (default is 0).

<a name="dataset.FrameAdapter"/>
### FrameAdapter ###
This adapters allows you to frame an image dataset to fixed dimensions. Images are first scaled while preserving aspect-ratio and then cropped to fit within the dimensions. Required options are:

  * `frame`(*) : Integer array of size 2 or 3 with the desired image dimensions.

<a name="dataset.AutoencoderAdapter"/>
### AutoencoderAdapter ###
This adapter sets the target of the dataset the same as the input (dropping any labels). This can be used to train auto-encoders, for example. No further options are required. This adapter can be activated using the `autoencode=true` option.

<a name="dataset.LabelAdapter"/>
### LabelAdapter ###
This adapter allows you to wrap a dataset in a new dataset with only a subset of the labels available. The other labels can be either ignored or be aggregated into one "other" class. Required options are:

  * `labels` : Integer array of the class indices that need to be preserved.
  * `other` : Boolean value indicating whether the remaining class indices need to be aggregated into an "other" label or not (default is `false`).

<a name="dataset.MaskToClassAdapter"/>
### MaskToClassAdapter ###
Changes a dataset aimed at segmentation (using masks) to binary classification, based on a threshold on the values of the mask. Required options are:

  * `center` : Boolean value indicating if we only need to look at the center pixel of the mask or look at the sum over all pixels (default is `false`).
  * `threshold` : Real value indicating the threshold above which the class is 1 instead of 0 (default is 0.5).

<a name="dataset.BinarizeAdapter"/>
### BinarizeAdapter ###
Changes the real-valued samples of a dataset to binary values, given a threshold on said values. This can be used to create black & white images from gray-scale images for example. Required options are:

  * `binarize`(*) : Real value indicating the threshold above which values will be set to 1, or 0 otherwise (default is 0.5).
