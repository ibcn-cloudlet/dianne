package be.iminds.iot.dianne.rl.learn.processors;

import java.util.EnumSet;
import java.util.Map;
import java.util.UUID;

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
	
	protected final Input targetInput;
	protected final Output targetOutput;
	
	protected final ExperiencePool pool;
	
	protected final float discountRate;
	
	private Tensor nextQ;
	private Tensor target;
	
	public TimeDifferenceProcessor(TensorFactory factory,
			Input input,
			Output output,
			Map<UUID, Trainable> toTrain,
			Input targetInput,
			Output targetOutput,
			ExperiencePool pool,
			Map<String, String> config) {
		super(factory, input, output, toTrain, pool, config);
		
		this.targetInput = targetInput;
		this.targetOutput = targetOutput;
		
		TargetListener listener = new TargetListener();
		this.targetInput.setMode(EnumSet.of(Mode.BLOCKING));
		this.targetOutput.addForwardListener(listener);
		
		this.pool = pool;
		
		// TODO set discountRate based on config
		discountRate = 0.99f;
	}

	@Override
	public void onForward(Tensor out, String... tags) {
		Tensor action = pool.getAction(index);
		float reward = pool.getReward(index);
		Tensor nextState = pool.getNextState(index);
		
		synchronized(this) {
			targetInput.input(nextState, ""+index);
			
			try {
				wait();
			} catch (InterruptedException e1) {}
		}
		
		target = out.copyInto(target);
		target.set(reward + discountRate * factory.getTensorMath().max(nextQ), factory.getTensorMath().argmax(action));
		
		Tensor e = criterion.error(out, target);
		error += e.get(0);
		
		output.backpropagate(criterion.grad(out, target), tags);
	}
	
	private class TargetListener implements ForwardListener {

		@Override
		public void onForward(Tensor output, String... tags) {
			synchronized(this) {
				nextQ = output;
				TimeDifferenceProcessor.this.notify();
			}
		}
		
	}
}
