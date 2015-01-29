package be.iminds.iot.dianne.nn.train.strategy;

import java.util.List;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.DatasetProcessor;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.tensor.Tensor;

public class StochasticGradient implements Trainer {

	private final int batchSize;
	private final int noEpochs;
	
	public StochasticGradient() {
		this(1,1);
	}
	
	public StochasticGradient(int batchSize, int noEpochs) {
		this.batchSize = batchSize;
		this.noEpochs = noEpochs;
	}
	
	@Override
	public void train(final Input input, final Output output, final List<Trainable> modules, 
			final Criterion criterion, final Dataset data) {
		System.out.println("Starting training");
		
		DatasetProcessor processor = new DatasetProcessor(input, output, data, true) {
			
			private int batch = 0;
			
			@Override
			protected void onForward(int index, Tensor out) {
				// forward done,  now back propagate
				Tensor mse = criterion.forward(out, data.getOutputSample(index));
				
				// Backward through output module
				output.backpropagate(criterion.backward(out, data.getOutputSample(index)));
			}
			
			@Override
			protected void onBackward(int index, Tensor in) {
				// back propagation done, now update params
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
			}
		};
		
		
		// repeat for a number of epochs
		for(int epoch=0;epoch<noEpochs;epoch++){			
			System.out.println(epoch);
			
			long t1 = System.currentTimeMillis();
			processor.process();
			long t2 = System.currentTimeMillis();
			System.out.println("Training time per sample: "+(double)(t2-t1)/(double)data.size()+" ms");
		}

	}

}
