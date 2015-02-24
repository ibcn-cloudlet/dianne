package be.iminds.iot.dianne.nn.train.strategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.AbstractTrainer;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.DatasetProcessor;
import be.iminds.iot.dianne.tensor.Tensor;

public class StochasticGradient extends AbstractTrainer {

	private final int batchSize;
	private final int noEpochs;
	private final float learningRate;
	private final float learningRateDecay;
	
	private int sample = 0;
	private int batch = 0;
	private int epoch = 0;
	private float error = 0;
	
	public StochasticGradient() {
		this(1,1, 0.01f,0);
	}
	
	public StochasticGradient(int batchSize, int noEpochs, float learningRate, float learningRateDecay) {
		this.batchSize = batchSize;
		this.noEpochs = noEpochs;
		this.learningRate = learningRate;
		this.learningRateDecay = learningRateDecay;
	}
	
	// TODO can now only do one call at a time cause mse/epoch/batch is counted in class
	@Override
	public void train(final Input input, final Output output, 
			final List<Trainable> modules, final List<Preprocessor> preprocessors,
			final Criterion criterion, final Dataset data) {
		System.out.println("Starting training");
		batch = 0;
		
		// first preprocess
		for(Preprocessor p : preprocessors){
			p.preprocess(data);
		}
		
		DatasetProcessor processor = new DatasetProcessor(input, output, data, true, true) {
			
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
					// anneal learning rate
					float lr = learningRate / (1 + batch*learningRateDecay);
					
					for(Trainable m : modules){
						m.updateParameters(lr);
						m.zeroGradParameters();
					}
					
					batch++;
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
			
			//Collections.shuffle(shuffle, new Random(0));
			
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
