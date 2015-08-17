package be.iminds.iot.dianne.nn.learn.processors;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class RandomBatchProcessor extends RandomSampleProcessor {

	private final int batchSize;
	
	public RandomBatchProcessor(TensorFactory factory, 
			Input input, 
			Output output, 
			Map<UUID, Trainable> toTrain, 
			Dataset dataset, 
			Map<String, String> config) {
		super(factory, input, output, toTrain, dataset, config);
		
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
		} catch (InterruptedException e1) {
		}
		
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

}
