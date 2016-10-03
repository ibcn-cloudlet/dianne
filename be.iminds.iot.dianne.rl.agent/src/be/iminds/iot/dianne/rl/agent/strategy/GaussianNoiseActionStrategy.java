package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.strategy.config.GaussianNoiseConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class GaussianNoiseActionStrategy implements ActionStrategy {
	
	private NeuralNetwork policy;
	private GaussianNoiseConfig config;
	
	private Tensor noise;

	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.policy = nns[0];
		this.config = DianneConfigHandler.getConfig(config, GaussianNoiseConfig.class);
		this.noise = new Tensor(env.actionDims());
	}

	@Override
	public AgentProgress processIteration(long i, Tensor state) throws Exception {
		Tensor action = policy.forward(state);
		
		noise.randn();
		
		double stdev = config.noiseMin + (config.noiseMax - config.noiseMin) * Math.exp(-i * config.noiseDecay);
		
		TensorOps.add(action, action, (float) stdev, noise);
		
		for(int a = 0; a < action.size(); a++) {
			float v = action.get(a);
			if(v < config.min)
				action.set(config.min, a);
			else if(v > config.max)
				action.set(config.max, a);
		}
		
		return new AgentProgress(i, action);
	}

}
