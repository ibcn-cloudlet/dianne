package be.iminds.iot.dianne.rl.learn.strategy;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepQConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class DeepDeterministicPolicyGradientStrategy implements LearningStrategy {

	protected DeepQConfig config;
	
	protected ExperiencePool pool;
	protected SamplingStrategy sampling;
	protected ExperiencePoolSample interaction;
	
	protected NeuralNetwork actor;
	protected NeuralNetwork targetActor;
	
	protected NeuralNetwork critic;
	protected NeuralNetwork targetCritic;
	
	protected Criterion criterion;
	protected GradientProcessor actorProcessor;
	protected GradientProcessor criticProcessor;
	
	protected UUID stateIn;
	protected UUID actionIn;
	protected UUID valueOut;
	
	protected Tensor targetValue;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 4)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 4)");
		
		this.actor = nns[0];
		this.targetActor = nns[1];
		this.critic = nns[2];
		this.targetCritic = nns[3];
		
		this.config = DianneConfigHandler.getConfig(config, DeepQConfig.class);
		this.sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		this.criterion = CriterionFactory.createCriterion(this.config.criterion);
		this.actorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, actor, config);
		this.criticProcessor = ProcessorFactory.createGradientProcessor(this.config.method, critic, config);
		
		NeuralNetworkInstanceDTO nndto = this.critic.getNeuralNetworkInstance();
		for(UUID iid : this.critic.getInputs().keySet()) {
			ModuleInstanceDTO mdto = nndto.modules.get(iid);
			String mname = mdto.module.properties.get("name");
			
			if(mname.equalsIgnoreCase("state"))
				this.stateIn = iid;
			else if(mname.equalsIgnoreCase("action"))
				this.actionIn = iid;
		}
		this.valueOut = this.critic.getOutput().getId();
		
		if(stateIn == null || actionIn == null || valueOut == null)
			throw new RuntimeException("Unable to select correct Input modules from network " + nndto.name);
		
		this.targetValue = new Tensor(1);
		
		while(pool.size() < this.config.minSamples){
			System.out.println("Experience pool has too few samples, waiting a bit to start learning...");
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				return;
			}
		}
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		actor.zeroDeltaParameters();
		critic.zeroDeltaParameters();
		
		float error = 0, value = 0;
		
		for(int b = 0; b < config.batchSize; b++) {
			int index = sampling.next();
			
			interaction = pool.getSample(interaction, index);
			
			Tensor state = interaction.getState();
			Tensor action = interaction.getAction();
			Tensor nextState = interaction.getNextState();
			float reward = interaction.getReward();

			UUID[] inputIds = new UUID[]{stateIn, actionIn};
			UUID[] outputIds = new UUID[]{valueOut};
			
			targetValue.fill(reward);
			
			if(!interaction.isTerminal) {
				Tensor nextAction = targetActor.forward(nextState);
				
				Tensor nextValue = targetCritic.forward(inputIds, outputIds, new Tensor[]{nextState, nextAction}).getValue().tensor;
				TensorOps.add(targetValue, targetValue, config.discount, nextValue);
			}
			
			Tensor currentValue = critic.forward(inputIds, outputIds, new Tensor[]{state, action}).getValue().tensor;
			
			value += currentValue.get(0);
			error += criterion.error(currentValue, targetValue).get(0);
			Tensor criticGrad = criterion.error(currentValue, targetValue);
			
			critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue();
			critic.accGradParameters();
			
			action = actor.forward(state);
			
			// TODO: if we could safely disable blocking we would only need to do the following...
//			p2 = critic.forward(actionIn, valueOut, action);
//			p2.getValue();
//			
//			criticGrad.fill(1);
//			
//			p2 = critic.backward(valueOut, actionIn, criticGrad);
//			Tensor actorGrad = p2.getValue().tensor;
			
			// ... unfortunately we currently still need to forward the state part of the critic as well.
			
			critic.forward(inputIds, outputIds, new Tensor[]{state, action}).getValue();
			criticGrad.fill(-1);
			Tensor actorGrad = critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue().tensors.get(actionIn);
			actor.backward(actorGrad);
			
			actor.accGradParameters();
		}
		
		batchAverage(actor, config.batchSize);
		batchAverage(critic, config.batchSize);
		
		error /= config.batchSize;
		value /= config.batchSize;
		
		actorProcessor.calculateDelta(i);
		criticProcessor.calculateDelta(i);
		
		actor.updateParameters();
		critic.updateParameters();
		
		return new QLearnProgress(i, error, value);
	}

	private static void batchAverage(NeuralNetwork nn, int batchSize) {
		nn.getTrainables().values().stream().forEach(m -> {
			Tensor deltaParams = m.getDeltaParameters();
			
			TensorOps.div(deltaParams, deltaParams, batchSize);
			
			// Set DeltaParameters to be sure in case of remote module instance
			m.setDeltaParameters(deltaParams);
		});
	}
}
