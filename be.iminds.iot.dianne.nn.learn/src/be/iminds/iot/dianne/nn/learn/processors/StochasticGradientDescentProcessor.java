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

public class StochasticGradientDescentProcessor extends AbstractProcessor implements ForwardListener, BackwardListener {

	// random generator
	private final Random rand = new Random(System.currentTimeMillis());

	// error criterion
	private final Criterion criterion;
	// learning rate
	private final float learningRate;
	// batch size
	private final int batchSize;
	
	// current error
	private float error = 0;
	// current index of the dataset that we process
	private int index;
	
	private CountDownLatch latch;
	
	public StochasticGradientDescentProcessor(TensorFactory factory, 
			Input input, 
			Output output, 
			Map<UUID, Trainable> toTrain, 
			Dataset dataset, 
			Map<String, String> config) {
		super(factory, input, output, toTrain, dataset, config);
		
		this.input.addBackwardListener(this);
		this.input.setMode(EnumSet.of(Mode.BLOCKING));
		this.output.addForwardListener(this);
		
		// TODO create criterion based on config
		this.criterion = new MSECriterion(factory);
		// TODO set learningRate based on config
		learningRate = 0.1f;
		// TODO get batchsize from config
		batchSize = 10;
	}
	
	
	@Override
	public float processNext() {
		error = 0;
		
		latch = new CountDownLatch(batchSize);
		
		forwardNext();

		// wait until done...
		try {
			latch.await();
		} catch (InterruptedException e1) {}
		
		applyLearningRate();
		
		return error/batchSize;
	}


	@Override
	public void onBackward(Tensor gradInput, String... tags) {
		accGradParameters();

		// if still more than one, forward next of the batch
		if(latch.getCount()>1){
			forwardNext();
		}
		
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
				m -> factory.getTensorMath().mul(m.getDeltaParameters(), m.getDeltaParameters(), -learningRate));
	}
	
	protected void forwardNext(){
		index = rand.nextInt(dataset.size());
		Tensor in = dataset.getInputSample(index);
		input.input(in, ""+index);
	}
}
