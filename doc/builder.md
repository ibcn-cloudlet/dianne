Builder: design and train your own neural network
=================================================

The DIANNE Builder allows you to design and train your own neural network models from within the browser. Begin by starting a DIANNE runtime with the Builder service, e.g. the `all` run configuration. Next, point your browser to:

	http://localhost:8080/dianne/ui/dianne.html

Or use `http://localhost:8080/dianne` for short. This should bring up the builder interface, consisting of a menu bar, a canvas and a toolbox with neural network building blocks, a.k.a. modules, on the left:

![Builder interface](figures/builder_1.tiff)

Designing your model is a two-step process. We will now recreate the MNIST classification network used in the [Getting started](http://dianne.intec.ugent.be/#gettingstarted) section. First, drag and drop the required modules from the toolbox onto the canvas. You can scroll down the toolbox list, and you can (un)collapse categories by clicking on the category title. Connect the modules together such that you start with an `Input` module and end with an `Output` module. In this case, we need two `Linear` modules, a `Sigmoid` module and a `LogSoftMax` module:

![The example MNIST neural network](figures/builder_2.tiff)

Next, we need to configure some of the modules in order to set their internal parameters correctly. You can configure modules by double clicking on them. Both `Linear` modules need to be configured with an input and output size. Since we are working on the MNIST dataset, which consists of 28x28 single channel images, the input of the first `Linear` should be 28 x 28 = 784. The output is the number of hidden neurons we would like, for example 50.

![Configure first Linear](figures/builder_3.tiff)

The second `Linear` module will then take 50 inputs, and the output will be the number of output classes, in the case of MNIST this is 10. Now deploy this neural network. When hitting the `Deploy` menu item, you will first be asked to save this neural network. Just give it a name you like:

![Save your neural network](figures/builder_4.tiff)

Once your network is deployed, click the `Learn` menu item to train your neural network. In this view, the module toolbox has been replaced by a learning toolbox, consisting off a `Trainer`, `Evaluator` and various datasets.

![Learning tab](figures/builder_5.tiff)

Start by adding an `MNIST` dataset and connecting this to your `Input`. In case the MNIST dataset is not showing up, please run `.\gradlew dataset -Pwhich=MNIST` to download it. Double clicking the dataset allows you to select how many samples will be used for training and how many will be used for testing.

![Dataset configuration](figures/builder_6.tiff)

To convince yourself that the neural network is actually not trained at all, connect an `Evaluator`, double click on it and click `evaluate`. This will evaluate the classification accuracy on the test set you just configured. From the confusion matrix you can see that the neural network is not yet trained.

![Untrained network](figures/builder_7.tiff)

Now also connect an `SGD Trainer` block to the `Output` and open the training dialog by double clicking on it. As we are doing classification, change the `Loss` option to `Negative Log Likelihood`. The other parameters' default values should suffice for this example, so then just click `Train`. Now the training procedure should start and you should see how the error decreases as training proceeds. 

![Training your neural network](figures/builder_8.tiff)

Click `Stop` when you are satisfied with the training. You can check the progress by running the `Evaluator` again. After a few minutes of training, your network should be able to reach over 90 percent accuracy. The confusion matrix should now highlight the diagonal meaning that the neural network performs as it is supposed to.

![Evaluating the trained network](figures/builder_9.tiff)

To convince yourself even further, you can go ahead to the `Run` tab and see the outputs for individual samples. You can download a pretrained version of this model form the DIANNE website using `./gradlew model -Pwhich=tutorial_1`.

Available modules
-----------------

In this example, only a handful of the available neural network modules were used, but many more are readily available in the DIANNE platform or will be added in the future. Besides modules implementing algebraic or element-wise operations, there are also *fork* and *join* modules, which allow your model to branch off and contain multiple paths. Moreover, there are *composite* modules, which allow other neural networks to be used as buildings blocks in a new model. ResNets, for example, can be constructed in this way. For more information on all the modules available in DIANNE, as well as there configuration parameters, please refer to the [module catalog on GitHub](modules.md).

Preconfigured datasets & models
-------------------------------

Besides the MNIST dataset and example model in the previous section, a number of other preconfigured datasets and models are available on the DIANNE [datasets page](http://dianne.intec.ugent.be/datasets/) and [models page](http://dianne.intec.ugent.be/models/) respectively. We will be adding more datasets and models from time to time. You can download these using the build tool:

	./gradlew dataset -Pwhich=<name>
	./gradlew model -Pwhich=<name>

**Note**: If you leave out the `which` option, all available datasets or models will be downloaded. Be aware this might take some time!

Add your own datasets
---------------------

For some dataset structures and file types, it is possible to add your own dataset with a simple configuration file, no additional coding required. For more information regarding datasets, see the [dataset documentation on GitHub](datasets.md).
