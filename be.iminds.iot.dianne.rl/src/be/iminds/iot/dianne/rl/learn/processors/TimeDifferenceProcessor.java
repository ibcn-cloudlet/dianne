package be.iminds.iot.dianne.rl.learn.processors;

import java.util.Map;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class TimeDifferenceProcessor extends StochasticGradientDescentProcessor {
	
	private final String[] logLabels = new String[]{"Q", "Target Q", "Error"};
	
	protected final NeuralNetwork target;
	
	protected final ExperiencePool pool;
	
	protected float discountRate = 0.99f;
	
	public TimeDifferenceProcessor(TensorFactory factory,
			NeuralNetwork nn,
			NeuralNetwork target,
			ExperiencePool pool,
			Map<String, String> config,
			DataLogger logger) {
		super(factory, nn, pool, config, logger);
		
		this.target = target;
		
		this.pool = pool;
		
		if(config.containsKey("discount"))
			discountRate = Float.parseFloat(config.get("discount"));
		
		System.out.println("TimeDifferenceRL");
		System.out.println("* discount rate = "+discountRate);
		System.out.println("---");
	}

	
	protected Tensor getGradOut(Tensor out, int index){
		
		Tensor action = pool.getAction(index);
		float reward = pool.getReward(index);
		Tensor nextState = pool.getNextState(index);
		
		float targetQ = 0;
		
		if(nextState==null){
			// terminal state
			targetQ = reward;
		} else {
			Tensor nextQ = target.forward(nextState, ""+index);
			targetQ = reward + discountRate * factory.getTensorMath().max(nextQ);
		}
		
		Tensor targetOut = out.copyInto(null);
		targetOut.set(targetQ, factory.getTensorMath().argmax(action));
		
		Tensor e = criterion.error(out, targetOut);
		error += e.get(0);
		
		if(logger!=null){
			logger.log("LEARN", logLabels, factory.getTensorMath().max(out), targetQ, e.get(0));
		}
		
		return criterion.grad(out, targetOut);
		
	}
}
