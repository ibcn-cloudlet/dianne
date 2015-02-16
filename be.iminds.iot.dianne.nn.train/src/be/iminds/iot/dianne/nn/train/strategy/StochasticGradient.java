package be.iminds.iot.dianne.nn.train.strategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.DatasetProcessor;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.tensor.Tensor;

public class StochasticGradient implements Trainer {

	private final int batchSize;
	private final int noEpochs;
	
	private int sample = 0;
	private int epoch = 0;
	private float error = 0;
	
	public StochasticGradient() {
		this(1,1);
	}
	
	public StochasticGradient(int batchSize, int noEpochs) {
		this.batchSize = batchSize;
		this.noEpochs = noEpochs;
	}
	
	// TODO can now only do one call at a time cause mse/epoch/batch is counted in class
	@Override
	public synchronized void train(final Input input, final Output output, final List<Trainable> modules, 
			final Criterion criterion, final Dataset data) {
		System.out.println("Starting training");
		
		DatasetProcessor processor = new DatasetProcessor(input, output, data, true) {
			
			@Override
			protected void onForward(int index, Tensor out) {
				// forward done,  now back propagate
				Tensor e = criterion.forward(out, data.getOutputSample(index));
				error+= e.get(0);
				
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
				sample++;
				if(sample % batchSize == 0){
					for(Trainable m : modules){
						m.updateParameters(0.5f);
						m.zeroGradParameters();
					}
				}
				
				if(sample % 500 == 0){
					error /= 500;
					System.out.println(error);
					notifyListeners();
					error = 0;
				}
			}
		};
		
		
		// repeat for a number of epochs
		for(epoch=0;epoch<noEpochs;epoch++){			
			System.out.println(epoch);
			sample = 0;
			
			long t1 = System.currentTimeMillis();
			processor.process();
			long t2 = System.currentTimeMillis();
			System.out.println("Training time per sample: "+(double)(t2-t1)/(double)data.size()+" ms");
		}

	}
	
	private List<TrainProgressListener> listeners = Collections.synchronizedList(new ArrayList<TrainProgressListener>());

	public void addProgressListener(TrainProgressListener l){
		this.listeners.add(l);
	}
	
	public void removeProgressListener(TrainProgressListener l){
		this.listeners.remove(l);
	}
	
	private void notifyListeners(){
		synchronized(listeners){
			for(TrainProgressListener l : listeners){
				l.onProgress(epoch, sample, error);
			}
		}
	}
}
