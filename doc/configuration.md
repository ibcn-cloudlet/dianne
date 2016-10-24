<a name="config"/>
# Configuration options #
This page gives an overview of the options that can be set when submitting jobs in DIANNE. Many of these options (and their default values) are defined in configuration classes, and the reader will be pointed to these where applicable.

  * [DianneCoordinator](#config.coordinator)
  * [Learner](#config.learner)
    * [LearningStrategies](#config.learningstrategies)
    * [Processors](#config.processors)
    * [Criterions](#config.criterions)
    * [SamplingStrategies](#config.samplingstrategies)
  * [Evaluator](#config.evaluator)
    * [EvaluationStrategies](#config.evaluationstrategies)
  * [Agent](#config.agent)
    * [ActionStrategies](#config.actionstrategies)
  
<a name="config.coordinator"/>
## DianneCoordinator ##
The `DianneCoordinator` is the central entity managing all jobs on the cluster. It maintains separate queues for each of the learn, evaluate and act jobs submitted through the Dashboard, command line or JSON RPC interface. When suitable idle `Learner`, `Evaluator` or `Agent` services, respectively, are available, the `DianneCoordinator` will deploy the required models and configure the respective services on their runtimes. Besides a number of options which are specific to the job type (listed below), following options are available:

  * `targets` : a comma-separated list of DIANNE runtime UUIDs on which the job needs to run. The job will only start when the required services are available on *all* the listed runtimes. Default is any available.
  * `targetCount` : the number of services that need to participate in the job. This needs to be set, for example, when using asycnhronous gradient descent using multiple `Learner` services. The default is 1.
  * `targetFilter` : a string describing a RFC 1960-based filter for the desired runtimes. Supported keys are `id` (runtime UUID), `name` (device hostname), `arch` (CPU architecture, e.g. `arch=arm`), `os` (operating system, e.g. `os=linux`) and `ip`.
  * `forceFree` : boolean value indicating whether a job should be the only one running on the selected nodes.
  * `strategy` : a string specifying the strategy to be used (exact interface/expected behavior depends on job type), either as a class name or as the source itself. In the case of source, the class will be compiled and distributed at runtime.

Following job types can be submitted to the coordinator:
* **LearnJob**: describes a learning job, such as the stop criteria. For learning parameters, please see [Learner](#config.learner) section. Following options are available:

  * `trainSet` : `minIndex,maxIndex` tuple indicating the range of the dataset to be used for learning. By default the entire dataset is used.
  * `validationSet` : `minIndex,maxIndex` tuple indicating the range of the dataset to be used for validation. Validation is performed after every synchronization with the repository by a suitable `Evaluator` service. By default no validation is performed.
  * `validationStrategy` : the `EvaluationStrategy` to be used when a `validationSet` is specified. See [EvaluationStrategies](#config.evaluationstrategies) for options. Default is `CriterionEvaluationStrategy`.
  * `maxIterations` : maximum number of weight updates calculated in this job. Default is -1, or infinite updates.
  * `errorThresholdWindow` : the number of iterations ago with which to compare the current error. Default is 10.
  * `miniBatchErrorThreshold` : the minimum decrease in minibatch error compared to `errorThresholdWindow` iterations ago. If lower than this threshold, training is stopped. Default is negative infinity.
  * `validationErrorThreshold` : the minimum decrease in validation error compared to `errorThresholdWindow` iterations ago. If lower than this threshold, training is stopped. Default is negative infinity.
  
* **EvaluationJob**: describes an evaluation job for testing the performance of a model. Please see the [Evaluator](#config.evaluator) section for options. One additional option is available:

  * `testSet` : `minIndex,maxIndex` tuple indicating the range of the dataset to be used for evaluation. By default the entire dataset is used.

* **ActJob**: sescribes a job of an `Agent` interacting with an `Environment`. Please see the [Agent](#config.agent) section for options.

<a name="config.learner"/>
## Learner ##
The `Learner` services in DIANNE allow you to train a (set of) model(s) using a dataset. In general, a `Learner` takes a (set of) deployed neural network instance(s) and dataset, calculates updates to the models' weights using a `LearningStrategy`, synchronizes those weights with the `Repository` and monitors the stopping criteria. Several specific `LearningStrategy` implementations are available for different learning tasks. Besides type-specific options, a list of common configuration options can be found in the [LearnerConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/config/LearnerConfig.java) configuration class.

<a name="config.learningstrategies"/>
### LearningStrategies ###
The `LearningStrategy` [interface](../be.iminds.iot.dianne.api/src/be/iminds/iot/dianne/api/nn/learn/LearningStrategy.java) defines two methods: a `setup(...)` method, which is called by the `Learner` to pass models, datasets, configuration options, etc., and a `processIteration(long i)` method that calculates (and applies) one update to the weights, and returns the loss over the processed sample or minibatch.

Following `LearningStrategy` implementations are currently available in DIANNE:
* **FeedForwardLearningStrategy**: the *default* `LearningStrategy`, used to train general feed-forward models (classifiers, auto-encoders,...) using (a flavor of) stochastic gradient descent. See [FeedForwardConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/strategy/config/FeedForwardConfig.java) for configuration options.
* **SimpleLearningStrategy**: a slimmed-down version of the `FeedForwardLearningStrategy`, as a bare-bones example of how to implement a `LearningStrategy`. Reuses the [FeedForwardConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/strategy/config/FeedForwardConfig.java) class for configuration options.
* **VariationalAutoEncoderLearningStrategy**: used for training VAEs [1]. This strategy takes two networks, an encoder or inference model, and a decoder or generative model. The output of the encoder are the parameters of a latent distribution from which samples are drown to pass to the decoder. Currently only factorized Gaussian-distributed latent variables are supported. Reuses the [FeedForwardConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/strategy/config/FeedForwardConfig.java) class for configuration options, as well as the following options:

  * `latentDims` : specifies the number of latent dimensions. Default is 1.
  * `sampleSize` : specifies the number of latent samples to be drawn for each gradient estimation. Default is 1.

* **GenerativeAdverserialLearningStrategy** : used for training GANs [2]. This strategy takes two networks, a generative model and a discriminative model and pits the two against each other. See [GenerativeAdverserialConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/strategy/config/GenerativeAdverserialConfig.java) for configuration options. 

There are also a number of strategies that are still highly experimental and will most likely change in following releases:
* **BPTTLearningStrategy**: use for training models which contain recurrent connections (consisting of `Memory` modules) using backpropagation-through-time. See [BPTTConfig](../be.iminds.iot.dianne.rnn/src/be/iminds/iot/dianne/rnn/learn/strategy/config/BPTTConfig.java) for configuration options.
* **DeepQLearningStrategy**: used for reinforcement learning with deep Q-learning. Requires two network instances of the Q-network, a target and a working instance. See [DeepQConfig](../be.iminds.iot.dianne.rl.learn/src/be/iminds/iot/dianne/rl/learn/strategy/config/DeepQConfig.java) for configuration options.
* **DeepDeterministicPolicyGradientStrategy**: used for reinforcement learning with DDPG. Requires 4 network instances: a target and working instance of both the actor and critic. Reuses the [DeepQConfig](../be.iminds.iot.dianne.rl.learn/src/be/iminds/iot/dianne/rl/learn/strategy/config/DeepQConfig.java) class for configuration options.

<a name="config.samplingstrategies"/>
### SamplingStrategies ###
To sample from `Dataset` services, DIANNE had a number of `SamplingStrategy` classes that generate random indices to collect samples:
* **SequentialSamplingStrategy**: cycles through the indices in order.
* **PermutationSamplingStrategy**: cycles through random permutations of the indices, shuffling the indices after every cycle.
* **UniformSamplingStrategy**: samples an index unformly.
* **WeightedUniformSamplingStrategy**: reweighs ranges of indices, but samples uniformly in those ranges. Can be used to balance datasets. See [WeightedUniformConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/sampling/config/WeightedUniformConfig.java) on how to configure.

<a name="config.criterions"/>
### Criterions ###
`Criterion` objects calculate the loss and the gradient with respect to that loss of a (batch of) output tensor(s) and a (batch of) target tensor(s). Following criterions are available:
* **AbsCriterion**: determines the L1 loss or absolute error.
* **BCECriterion.java**: determines the binary cross entropy loss.
* **GaussianKLDivCriterion**: determines the KL-divergence between two factorized Gaussian distributions. Tensors first list the means, then the standard deviations.
* **MSECriterion**: calculates the L2 loss or mean squared error.
* **NLLCriterion**: calculates the negative log likelihood. Tensors can either be log probabilities or regular probabilities.

<a name="config.processors"/>
### Processors ###
To determine weight updates based on weight gradients, DIANNE has the concept of `GradientProcessor`s. These objects take the (accumulated) gradients to the model's weights and transform them in some way, e.g. multiply them with a learning rate. A first group of processors is set using the `method` option of the `Learner`. Following processors perform optimization using a variant of gradient descent.
* **StochasticGradientDescentProcessor**: performs (vanilla) stochastic gradient descent, possibly with weight decay. See [SGDConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/SGDConfig.java) for options. This is also the default when no `method` option is passed.
* **RMSpropProcessor**: performs RMSprop optimization [3]. See [RMSpropConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/RMSpropConfig.java) for options.
* **AdagradProcessor**: performs Adagrad optimization [4]. See [AdagradConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdagradConfig.java) for options.
* **AdadeltaProcessor**: performs Adadelta optimization [5]. See [AdadeltaConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdadeltaConfig.java) for options.
* **AdamProcessor**: performs Adam optimization [6]. See [AdamConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdamConfig.java) for options.

A second group of processors *decorates* the ones specified by the `method` option. These processors are only added if their options are explicitly set.
* **Regularizationprocessor**: performs L2 weight decay. See [RegularizationConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/RegularizationConfig.java) for options.
* **MomentumProcessor**: performs (vanilla) momentum. See [MomentumConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/MomentumConfig.java) for options.
* **NesterovMomentumProcessor**: performs Nesterov momentum [7]. See [NesterovConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/NesterovConfig.java) for options.

<a name="config.evaluator"/>
## Evaluator ##
The task of the `Evaluator` services in DIANNE is to assess the performance of a model. By separating an evaluation task from a learning task, we can evaluate models on specific platforms on which training itself is not feasible and report performance measures besides the loss. Like with the `Learner` services, an `Evaluator` takes a deployed model, a dataset an `EvaluationStrategy` object to perform the evaluation itself and a number of configuration options. Besides strategy-specific options, a list of common configuration options can be found in the [EvaluatorConfig](../be.iminds.iot.dianne.nn.eval/src/be/iminds/iot/dianne/nn/eval/config/EvaluatorConfig.java) configuration class.

<a name="config.evaluationstrategies"/>
### EvaluationStrategies ###
Following types of `EvaluatorStrategy` implementations are currently available in DIANNE:
* **ClassificationEvaluationStrategy** : will calculate the classification accuracy on the given dataset, as well as report the top-k error and a confusion matrix. this is the default `EvaluationStrategy`.
* **CriterionEvaluationStrategy**: given a criterion, this strategy will report the average loss of that criterion on the selected dataset.
* **VariationalAutoEncoderEvaluationStrategy**: evaluate a VAE [1] by reporting the mean reconstruction error given the maximum likelihood latent variables for each datapoint. When the `includeOutputs` option is set to `true`, reported outputs are those latent variables. Currently only factorized Gaussian-distributed latent variables are supported.

All strategies currently use the [EvaluationStrategyConfig](../be.iminds.iot.dianne.nn.eval/src/be/iminds/iot/dianne/nn/eval/strategy/config/EvaluationStrategyConfig.java) class for configuration options.

<a name="config.agent"/>
## Agent ##
`Agent` services are used to interact with an `Environment` service and push experience to an `ExperiencePool`, which can later be used for training reinforcement learning models. This agent feeds observations from the environment through a neural network model and generates an action based on the output. For configuration, please see the [AgentConfig](../be.iminds.iot.dianne.rl.agent/src/be/iminds/iot/dianne/rl/agent/config/AgentConfig.java) configuration class.

**Note:** Reinforcement learning in DIANNE is still a work in progress, please have a look on the `master` branch for new features.

<a name="config.actionstrategies"/>
### ActionStrategies ###
`ActionStrategy` objects determine how an agent should interpret the output tensor of a model (e.g. deep neural network) in order to select an action to perform on the `Environment`, as well as trade-off exploration versus exploitation. Following strategies are currently available:
* **GreedyActionStrategy**: performs epsilon-greedy exploration with optional epsilon decay. See [GreedyConfig](../be.iminds.iot.dianne.rl.agent/src/be/iminds/iot/dianne/rl/agent/strategy/config/GreedyConfig.java) for configuration options. This is the default `ActionStrategy`.
* **BoltzmannStrategy**: transforms the output of the model into a probability distribution over (discrete) actions and samples from it. An optional (decaying) temperature allows trading off between exploration and exploitation. See [BoltzmannConfig](../be.iminds.iot.dianne.rl.agent/src/be/iminds/iot/dianne/rl/agent/strategy/config/BoltzmannConfig.java) for configuration options.
* **GaussianNoiseActionStrategy**: adds zero-mean Gaussian noise to (continuous) actions generated by the network. Action values are clipped afterwards to ensure they remain in a valid domain. See [GaussianNoiseConfig](../be.iminds.iot.dianne.rl.agent/src/be/iminds/iot/dianne/rl/agent/strategy/config/GaussianNoiseConfig.java) for configuration options.
* **RandomActionStrategy**: generates random continuous actions in `[-1,1]` and hence does not require a model.
* **ManualActionStrategy**: a special strategy for taking manual control of the `Agent`, and generate *human* experience. See the [Pong environment](../be.iminds.iot.dianne.rl.environment.pong/src/be/iminds/iot/dianne/rl/environment/pong) for an example.

<a name="config.references"/>
## References ##
[1] D. P. Kingma and M. Welling (2013). Auto-Encoding Variational Bayes. Retrieved from [http://arxiv.org/abs/1312.6114](http://arxiv.org/abs/1312.6114)

[2] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio (2014). Generative Adversarial Networks. Retrieved from [http://arxiv.org/abs/1406.2661](http://arxiv.org/abs/1406.2661)

[3] http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

[4] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html

[5] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from [http://arxiv.org/abs/1212.5701](http://arxiv.org/abs/1212.5701)

[6] Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.

[7] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.
