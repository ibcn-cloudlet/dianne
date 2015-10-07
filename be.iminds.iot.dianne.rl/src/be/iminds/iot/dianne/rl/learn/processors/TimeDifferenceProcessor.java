package be.iminds.iot.dianne.rl.learn.processors;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
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
			DataLogger logger,
			ExperiencePool pool, 
			SamplingStrategy s,
			Criterion c,
			float learningRate,
			int batchSize,
			float discount) {
		super(factory, nn, logger, pool, s, c, learningRate, batchSize);
		
		this.target = target;
		this.pool = pool;
		this.discountRate = discount;
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
