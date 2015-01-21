package be.iminds.iot.dianne.nn.train.strategy;

import java.util.List;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.InputListener;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.OutputListener;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.tensor.Tensor;

public class StochasticGradient implements Trainer {

	private int batchSize = 10;
	private int noEpochs = 1;
	
	private int index = 0;
	private int batch = 0;
	
	public StochasticGradient() {	
	}
	
	public StochasticGradient(int batchSize, int noEpochs) {
		this.batchSize = batchSize;
		this.noEpochs = noEpochs;
	}
	
	@Override
	public void train(final Input input, final Output output, final List<Trainable> modules, 
			final Criterion criterion, final Dataset data) {
		System.out.println("Starting training");

		final OutputListener outputListener = new OutputListener() {
			@Override
			public void onForward(Tensor out) {
				// forward done,  now back propagate
				Tensor mse = criterion.forward(out, data.getOutputSample(index));
				
				// Backward through output module
				output.backpropagate(criterion.backward(out, data.getOutputSample(index)));
			}
		};
		output.addOutputListener(outputListener);
		
		
		final InputListener inputListener = new InputListener() {
			@Override
			public void onBackward(Tensor gradInput) {
				// back propagation done, now update params if needed and do next item
				// accGradParameters for all trainable modules
				for(Trainable m : modules){
					m.accGradParameters();
				}
			
				// updateParameters after batch
				batch++;
				if(batch > batchSize){
					for(Trainable m : modules){
						m.updateParameters(0.5f);
						m.zeroGradParameters();
					}
					batch = 0;
				}
				
				
				index++;
				if(index < data.size()){
					Tensor in = data.getInputSample(index);
					input.input(in);
				} else {
					synchronized(StochasticGradient.this){
						StochasticGradient.this.notifyAll();
					}
				}
			}
		};
		input.addInputListener(inputListener);
		
		
		// repeat for a number of epochs
		for(int epoch=0;epoch<noEpochs;epoch++){			
			System.out.println(epoch);
			
			long t1 = System.currentTimeMillis();
			
			index = 0;
			batch = 0;
			
			// Read first sample
			Tensor in = data.getInputSample(index);
			input.input(in);;
			
			synchronized(StochasticGradient.this){
				try {
					StochasticGradient.this.wait();
				} catch (InterruptedException e) {
				}
			}
			
			
			long t2 = System.currentTimeMillis();
			System.out.println("Training time per sample: "+(double)(t2-t1)/(double)data.size()+" ms");
		}
		// repeat
			
		output.removeOutputListener(outputListener);
		input.removeInputListener(inputListener);
	}

}
