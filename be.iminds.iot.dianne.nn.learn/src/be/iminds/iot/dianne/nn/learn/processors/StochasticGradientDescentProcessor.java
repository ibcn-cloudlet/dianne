package be.iminds.iot.dianne.nn.learn.processors;

import java.util.EnumSet;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.learn.criterion.NLLCriterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class StochasticGradientDescentProcessor extends AbstractProcessor implements ForwardListener, BackwardListener {

	// error criterion
	protected Criterion criterion;
	// learning rate
	protected float learningRate = 0.01f;
	// batch size
	protected int batchSize = 10;
	
	// random generator
	private final Random rand = new Random(System.currentTimeMillis());
	
	// current error
	protected float error;
	// current index of the dataset that we process
	protected int index;
	
	// count down latch
	private CountDownLatch latch;
	
	public StochasticGradientDescentProcessor(TensorFactory factory, 
			Input input, 
			Output output, 
			Map<UUID, Trainable> toTrain, 
			Dataset dataset, 
			Map<String, String> config,
			DataLogger logger) {
		super(factory, input, output, toTrain, dataset, config, logger);
		
		this.input.addBackwardListener(this);
		this.input.setMode(EnumSet.of(Mode.BLOCKING));
		this.output.addForwardListener(this);
		
		this.criterion = new MSECriterion(factory);
		String c = config.get("criterion");
		if(c!=null){
			if(c.equals("NLL")){
				criterion = new NLLCriterion(factory);
			} else if(c.equals("MSE")){
				criterion = new MSECriterion(factory);
			}
		}

		String l = config.get("learningRate");
		if(l!=null){
			learningRate = Float.parseFloat(l);
		}
		
		String b = config.get("batchSize");
		if(b!=null){
			batchSize = Integer.parseInt(b);
		}
		
		System.out.println("StochasticGradientDescent");
		System.out.println("* criterion = "+criterion.getClass().getName());
		System.out.println("* learningRate = "+learningRate);
		System.out.println("* batchSize = "+batchSize);
		System.out.println("---");
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
