package be.iminds.iot.dianne.nn.train.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.train.DatasetProcessor;
import be.iminds.iot.dianne.api.nn.train.Evaluation;
import be.iminds.iot.dianne.api.nn.train.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class ArgMaxEvaluator implements Evaluator {

	protected final TensorFactory factory;
	
	public ArgMaxEvaluator(TensorFactory factory) {
		this.factory = factory;
	}
	
	private int sample = 0;
	
	@Override
	public synchronized Evaluation evaluate(Input input, Output output, final Dataset data) {
		final Tensor confusion = factory.createTensor(output.getOutputLabels().length, output.getOutputLabels().length);
		confusion.fill(0.0f);
		
		final DatasetProcessor processor = new DatasetProcessor(input, output, data, false, false) {
			
			@Override
			protected void onForward(int index, Tensor out) {
				int predicted = factory.getTensorMath().argmax(out);
				int real = factory.getTensorMath().argmax(data.getOutputSample(index));
					
				confusion.set(confusion.get(real, predicted)+1, real, predicted);
				
				sample++;
				
				if(sample % 500 == 0)
					notifyListeners(confusion);
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

	private List<EvalProgressListener> listeners = Collections.synchronizedList(new ArrayList<EvalProgressListener>());
	
	public void addProgressListener(EvalProgressListener l){
		this.listeners.add(l);
	}
	
	public void removeProgressListener(EvalProgressListener l){
		this.listeners.remove(l);
	}
	
	private void notifyListeners(Tensor confusionMatrix){
		synchronized(listeners){
			for(EvalProgressListener l : listeners){
				l.onProgress(confusionMatrix);
			}
		}
	}

}
