package be.iminds.iot.dianne.nn.learn.processors;

import java.util.EnumSet;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class RandomSampleProcessor extends AbstractProcessor implements ForwardListener, BackwardListener {

	// random generator
	protected final Random rand = new Random(System.currentTimeMillis());

	// error criterion
	protected Criterion criterion;
	// learning rate
	protected float learningRate = 0.1f;
	
	
	// current error
	protected float error = 0;
	// current index of the dataset that we process
	protected int index;
	
	protected CountDownLatch latch;
	
	public RandomSampleProcessor(TensorFactory factory, 
			Input input, 
			Output output, 
			Map<UUID, Trainable> toTrain, 
			Dataset dataset, 
			Map<String, String> config) {
		super(factory, input, output, toTrain, dataset);
		this.input.addBackwardListener(this);
		this.input.setMode(EnumSet.of(Mode.BLOCKING));
		this.output.addForwardListener(this);
		// TODO create criterion based on config
		this.criterion = new MSECriterion(factory);
		// TODO set learningRate based on config
	}
	
	
	@Override
	public float processNext() {
		error = 0;
		
		latch = new CountDownLatch(1);
		
		forwardNext();

		// wait until done...
		try {
			latch.await();
		} catch (InterruptedException e1) {
		}
		
		applyLearningRate();
		
		return error;
	}


	@Override
	public void onBackward(Tensor gradInput, String... tags) {
		accGradParameters();

		// notify
		latch.countDown();
	}

	@Override
	public void onForward(Tensor out, String... tags) {
		Tensor e = criterion.error(out, dataset.getOutputSample(index));
		error += e.get(0);
		
		// Backward through output module
		output.backpropagate(criterion.grad(out, dataset.getOutputSample(index)), tags);
	}

	protected void accGradParameters(){
		// acc gradParameters
		toTrain.values().stream().forEach(m -> m.accGradParameters());
	}
	
	protected void applyLearningRate(){
		// multiply with learning rate
		toTrain.values().stream().forEach(
				m -> factory.getTensorMath().mul(m.getGradParameters(), m.getGradParameters(), learningRate));
	}
	
	protected void forwardNext(){
		index = rand.nextInt(dataset.size());
		Tensor in = dataset.getInputSample(index);
		input.input(in, ""+index);
	}
}
