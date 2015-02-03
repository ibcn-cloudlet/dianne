package be.iminds.iot.dianne.nn.train.eval;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.train.DatasetProcessor;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ArgMaxEvaluator implements Evaluator {

	protected final TensorFactory factory;
	
	public ArgMaxEvaluator(TensorFactory factory) {
		this.factory = factory;
	}
	
	@Override
	public Evaluation evaluate(Input input, Output output, final Dataset data) {
		final Tensor confusion = factory.createTensor(data.outputSize(), data.outputSize());
		confusion.fill(0.0f);
		
		final DatasetProcessor processor = new DatasetProcessor(input, output, data, false) {
			
			@Override
			protected void onForward(int index, Tensor out) {
				int predicted = factory.getTensorMath().argmax(out);
				int real = factory.getTensorMath().argmax(data.getOutputSample(index));
					
				confusion.set(confusion.get(real, predicted)+1, real, predicted);
			}
			
			@Override
			protected void onBackward(int index, Tensor in) {
				// will not be called!
			}
		};
		
		long t1 = System.currentTimeMillis();
		processor.process();	
		long t2 = System.currentTimeMillis();
		System.out.println("Forward time per sample: "+(double)(t2-t1)/(double)data.size()+" ms");
		
		return new Evaluation(factory, confusion);
	}


}
