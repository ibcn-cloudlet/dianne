<a name="config"/>
# Configuration options #
This page gives an overview of the options that can be set when submitting jobs in DIANNE. Many of these options (and their default values) are defined in configuration classes, and the reader will be pointed to these where applicable.

  * [DianneCoordinator](#config.DianneCoordinator)
  * [Learners](#config.learners)
    * [Processors](#config.processors)
  * [Evaluators](#config.evaluators)
  * [Agents](#config.agents)
    * [ActionStrategies](#config.strategies)
  
<a name="dataset.DianneCoordinator"/>
## DianneCoordinator ##
The `DianneCoordinator` is the central entity managing all jobs on the cluster. It maintains separate queues for each of the learn, evaluate and act jobs submitted through the Dashboard or JSON RPC interface. When suitable idle `Learner`, `Evaluator` and `Agent` services, respectively, are available, the `DianneCoordinator` will deploy the required models and configure the services on their respective runtimes. Besides a number of options which are specific to the job type (listed below), following options are available:

  * `targets` : A comma-separated list of DIANNE runtime UUIDs on which the job needs to run. The job wil only start when the required services are available on *all* the listed runtimes. Default is any available.
  * `targetCount` : The number of services that need to participate in the job. This needs to be set, for example, when using asycnhronous gradient descent using multiple `Learner` services. The default is 1.
  * `targetFilter` : A string describing a RFC 1960-based filter for the desired runtimes. Supported keys are `id` (runtime UUID), `name` (device hostname), `arch` (CPU architecture, e.g. `arch=arm`), `os` (operating system, e.g. `os=linux`) and `ip`.

Following job types can be submitted to the coordinator:
* **LearnJob**: Describes a learning job, such as the stop criteria. For learning parameters, please see [Learners](#config.learners) section. Following options are available:

  * `trainSet` : `minIndex,maxIndex` couple indicating the range of the dataset to be used for learning. By default the entire dataset is used.
  * `validationSet` : `minIndex,maxIndex` couple indicating the range of the dataset to be used for validation. Validation is performed after every synchronization with the repository by a suitable `Evaluator` service. By default no validation is performed.
  * `maxIterations` : Maximum number of updates to the repository for this job. Default is -1, or infinite updates.
  * `errorThresholdWindow` : The number of iterations ago with which to compare the current error. Default is 10.
  * `miniBatchErrorThreshold` : The minimum decrease in minibatch error compared to `errorThresholdWindow` iterations ago. If lower than this threshold, training is stopped. Default is negative infinity.
  * `validationErrorThreshold` : The minimum decrease in validation error compared to `errorThresholdWindow` iterations ago. If lower than this threshold, training is stopped. Default is negative infinity.
  
* **EvaluationJob**: Describes an evaluation job for testing the performance of a model. Please see the [Evaluators](#config.evaluators) section for options.
* **ActJob**: Describes a job of an `Agent` interacting with an `Environment`. Please see the [Agents](#config.agents) section for options.

<a name="config.learners"/>
## Learners ##
The `Learner` services in DIANNE allow you to train a model using a dataset. In general, a `Learner` takes a deployed neural network instance and datasets, and generates updates to the weights. Several specific `Learner`s are available for different learning tasks. Currently the type of `Learner` is determined automatically. Besides type-specific options, a list of common configuration options can be found in the [LearnerConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/config/LearnerConfig.java) configuration class. Following types of `Learner` services are currently available in DIANNE:
* **FeedForwardLearner**: The *default* `Learner`, used to train feed-forward models using (a flavor of) stochastic gradient descent. See [FeedForwardLearnerConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/config/FeedForwardLearnerConfig.java) for configuration options.
* **RecurrentLearner**: Use for training models which contain recurrent connections (consisting of `Memory` modules) using backpropagation-through-time. See [RecurrentLearnerConfig](../be.iminds.iot.dianne.rnn/src/be/iminds/iot/dianne/rnn/learn/config/RecurrentLearnerConfig.java) for configuration options. **Note: experimental!**
* **DeepQLearner**: Used for reinforcement learning using deep Q-learning. Will be automatically selected if there is a `environment` configuration option or the `dataset` option refers to an `ExperiencePool` service. Two model instances will be deployed for this learner. See [DeepQLearnerConfig](../be.iminds.iot.dianne.rl/src/be/iminds/iot/dianne/rl/learn/config/DeepQLearnerConfig.java) for configuration options. **Note: experimental!**

<a name="config.processors"/>
### Processors ###
In order to reuse update rules between different learners, DIANNE has the concept of `GradientProcessor`s. These objects take the (accumulated) gradients to the model's weights and transform them in some way, e.g. multiply them with a learning rate. A first group of processors is set using the `method` option of the `Learner`. Following processors perform optimization using a variant of gradient descent.
* **StochasticGradientDescentProcessor**: Performs (vanilla) stochastic gradient descent, possibly with weigth decay. See [SGDConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/SGDConfig.java) for options. This is also the default when no `method` option is passed.
* **RMSpropProcessor**: Performs RMSprop optimization [1]. See [RMSpropConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/SGDConfig.java) for options.
* **AdagradProcessor**: Performs Adagrad optimization [2]. See [AdagradConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdagradConfig.java) for options.
* **AdadeltaProcessor**: Performs Adadelta optimization [3]. See [AdadeltaConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdadeltaConfig.java) for options.
* **AdamProcessor**: Performs Adam optimization [4]. See [AdamConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/AdamConfig.java) for options.

A second group of processors *decorates* the ones specified by the `method` option. These processors are only added if their options are explicitly set.
* **Regularizationprocessor**: Performs L2 weight decay. See [RegularizationConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/RegularizationConfig.java) for options.
* **MomentumProcessor**: Performs (vanilla) momentum. See [MomentumConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/MomentumConfig.java) for options.
* **NesterovMomentumProcessor**: Performs Nesterov momentum [5]. See [NesterovConfig](../be.iminds.iot.dianne.nn.learn/src/be/iminds/iot/dianne/nn/learn/processors/config/NesterovConfig.java) for options.

<a name="config.evaluators"/>
## Evaluators ##
The task of the `Evaluator` services in DIANNE is to assess the performance of a model. By separating an evaluation task from a learning task, we can evaluate models on specific platforms on which training itself is not feasible and report performance measures besides strict error. Like with the `Learner` services, different types of `Evaluator`s exist which are selected automatically based on the used dataset, model and options. Besides type-specific options, a list of common configuration options can be found in the [EvaluatorConfig](../be.iminds.iot.dianne.nn.eval/src/be/iminds/iot/dianne/nn/eval/config/EvaluatorConfig.java) configuration class. `Evaluator`s are only available for feed-forward models at the moment. Following types of `Evaluator` services are currently available in DIANNE:
* **CriterionEvaluator**: Given a criterion, this `Evaluator` will report the average error of that criterion on the selected dataset. See [CriterionEvaluatorConfig](../be.iminds.iot.dianne.nn.eval/src/be/iminds/iot/dianne/nn/eval/config/CriterionEvaluatorConfig.java) for configuration options. This is the default `Evaluator`.
* **ClassificationEvaluator** : Will calculate the classification accuracy on the given dataset, as well as report a confusion matrix. This `Evaluator` is selected automatically selected when the dataset has labels. No further configuration options are available.

<a name="config.agents"/>
## Agents ##
`Agent` services are used to interact with an `Environment` service and push experience to an `ExperiencePool`, which can later be used for training reinforcement learning strategies. Currently only one type of `Agent` service is available, the `DeepRLAgent`. This agent feeds observations from the environment through a neural network model and generates an action based on the output. For configuration, please see the [AgentConfig](../be.iminds.iot.dianne.rl/src/be/iminds/iot/dianne/rl/agent/config/AgentConfig.java) configuration class.

**Note:** Reinforcement learning in DIANNE is still a work in progress, please have a look on the `master` branch for new features.

<a name="config.strategies"/>
### ActionStrategies ###
`ActionStrategy` objects determine how an agent should interpret the output tensor of a model (e.g. deep neural network) in order to select an action to perform on the `Environment`, as well as trade-off exploration versus exploitation. Following strategies are currently available:
* **GreedyActionStrategy**: Performs epsilon-greedy exploration with optional epsilon decay. See [GreedyConfig](../be.iminds.iot.dianne.rl/src/be/iminds/iot/dianne/rl/agent/strategy/config/GreedyConfig.java) for configuration options. This is the default `ActionStrategy`.
* **BoltzmannStrategy**: Transforms the output of the model into a probability distribution over (discrete) actions and samples from it. An optional (decaying) temperature allows trading off between exploration and exploitation. See [BoltzmannConfig](../be.iminds.iot.dianne.rl/src/be/iminds/iot/dianne/rl/agent/strategy/config/BoltzmannConfig.java) for configuration options.
* **ManualActionStrategy**: A special strategy for taking manual control of the `Agent`, and generate *human* experience. See the [Pong environment](../be.iminds.iot.dianne.rl/src/be/iminds/iot/dianne/rl/pong) for an example.

<a name="config.references"/>
## References ##
[1] http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

[2] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html

[3] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701

[4] Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations, 1–13.

[5] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.
