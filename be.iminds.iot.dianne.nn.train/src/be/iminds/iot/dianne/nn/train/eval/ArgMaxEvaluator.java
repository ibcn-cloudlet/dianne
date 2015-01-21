package be.iminds.iot.dianne.nn.train.eval;

import java.util.concurrent.CountDownLatch;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.OutputListener;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ArgMaxEvaluator implements Evaluator {

	protected static final TensorFactory factory = TensorFactory.getFactory(TensorFactory.TensorType.JAVA);
	
	int index = 0;
	
	@Override
	public Evaluation evaluate(Input input, Output output, final Dataset data) {
		final Tensor confusion = factory.createTensor(data.outputSize(), data.outputSize());
		confusion.fill(0.0f);
		index = 0;
		
		// Latch to wait
		final CountDownLatch latch = new CountDownLatch(data.size());
		
		// Add outputlistener
		final OutputListener outputListener = new OutputListener() {
			
			@Override
			public void onForward(Tensor output) {	
				// evaluate
				Tensor out = data.getOutputSample(index);
					
				int predicted = factory.getTensorMath().argmax(output);
				int real = factory.getTensorMath().argmax(out);
					
				confusion.set(confusion.get(real, predicted)+1, real, predicted);
				
				latch.countDown();
				
				// launch next item
				index++;
				if(index < data.size()){
					Tensor in = data.getInputSample(index);
					input.input(in);
				}
			}
		};
		output.addOutputListener(outputListener);
		
		long t1 = System.currentTimeMillis();
		
		// Forward first sample
		Tensor in = data.getInputSample(index);
		input.input(in);
		
		try {
			latch.await();
		} catch (InterruptedException e) {
		}
			
		long t2 = System.currentTimeMillis();
		System.out.println("Forward time per sample: "+(double)(t2-t1)/(double)data.size()+" ms");
		
		output.removeOutputListener(outputListener);
		
		return new Evaluation(confusion);
	}


}
