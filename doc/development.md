Development
===========

Using the DIANNE APIs
---------------------

Since DIANNE is built on top of OSGi/AIOLOS, all services that are used internally are also available for 3rd party applications. The most interesting OSGi services are the following:

- `DiannePlatform`: offers an API to deploy neural networks from the repository, or deploy individual neural network modules

- `Dianne`: allows you to get a `NeuralNetwork` reference to a neural network instance deployed via `DiannePlatform`. This offers a clean API to interact with a neural network instance, providing methods to perform a forward or backward pass, get or set the parameters, query all modules, etc..

- `DianneCoordinator`: is a more high level interface that allows you to submit learn, evaluation and act tasks to DIANNE, without having to deal with all the internals of the framework. This interface can also be exposed via JSON RPC.

For more detailed information, check the [`be.iminds.iot.dianne.api`](https://github.com/ibcn-cloudlet/dianne/tree/v0.4.0/be.iminds.iot.dianne.api) project.

Developing for DIANNE
---------------------

To develop and build your own OSGi bundles, the [BND](http://www.aqute.biz/Bnd/Bnd) tool is used. 
This is included in the git repository, so no additional download is required. Although any code editor suffices to develop for DIANNE, we recommend the following setup:

- [Eclipse](http://www.eclipse.org/downloads/)

- [BNDTools](http://bndtools.org/) plugin for Eclipse
	The BNDTools plugin allows you to easily view and edit .bnd files used for configuring
	and building OSGi bundles. This plugin also allows you to create run configurations to
	easily test the framework on your local computer.

	BNDTools can be downloaded from the marketplace inside Eclipse. (Help > Eclipse Marketplace... > Search for BNDTools)

	A nice tutorial on how to develop OSGi applications using BNDTools is available from
	their [website](http://bndtools.org/tutorial.html).

Once configured, you should be able to import all bundle projects in your Eclipse workspace as well as add your own. You can also modify and run the configurations in the `tools` project from within Eclipse.
