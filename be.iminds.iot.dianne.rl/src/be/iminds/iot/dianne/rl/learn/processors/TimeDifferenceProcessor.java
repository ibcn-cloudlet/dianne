package be.iminds.iot.dianne.rl.learn.processors;

import java.util.EnumSet;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class TimeDifferenceProcessor extends StochasticGradientDescentProcessor {
	
	private final String[] logLabels = new String[]{"Q", "Target Q", "Error"};
	
	protected final Input targetInput;
	protected final Output targetOutput;
	
	protected final ExperiencePool pool;
	
	protected float discountRate = 0.99f;
	
	private Tensor nextQ;
	private Tensor target;
	
	public TimeDifferenceProcessor(TensorFactory factory,
			Input input,
			Output output,
			Map<UUID, Trainable> toTrain,
			Input targetInput,
			Output targetOutput,
			ExperiencePool pool,
			Map<String, String> config,
			DataLogger logger) {
		super(factory, input, output, toTrain, pool, config, logger);
		
		this.targetInput = targetInput;
		this.targetOutput = targetOutput;
		
		this.targetInput.setMode(EnumSet.of(Mode.BLOCKING));
		this.targetOutput.addForwardListener(new TargetListener());
		
		this.pool = pool;
		
		if(config.containsKey("discount"))
			discountRate = Float.parseFloat(config.get("discount"));
		
		System.out.println("TimeDifferenceRL");
		System.out.println("* discount rate = "+discountRate);
		System.out.println("---");
	}

	@Override
	public void onForward(Tensor out, String... tags) {
		Tensor action = pool.getAction(index);
		float reward = pool.getReward(index);
		Tensor nextState = pool.getNextState(index);
		
		float targetQ = 0;
		
		if(nextState==null){
			// terminal state
			targetQ = reward;
		} else {
			synchronized(this) {
				targetInput.input(nextState, ""+index);
				
				try {
					wait();
				} catch (InterruptedException e) {}
			}
			
			target = out.copyInto(target);
			
			targetQ = reward + discountRate * factory.getTensorMath().max(nextQ);
		}
		
		target.set(targetQ, factory.getTensorMath().argmax(action));
		
		Tensor e = criterion.error(out, target);
		error += e.get(0);
		
		if(logger!=null){
			logger.log("LEARN", logLabels, factory.getTensorMath().max(out), targetQ, e.get(0));
		}
		
		output.backpropagate(criterion.grad(out, target), tags);
	}
	
	private class TargetListener implements ForwardListener {

		@Override
		public void onForward(Tensor output, String... tags) {
			synchronized(TimeDifferenceProcessor.this) {
				nextQ = output;
				TimeDifferenceProcessor.this.notify();
			}
		}
		
	}
}
