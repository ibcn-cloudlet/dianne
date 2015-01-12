package be.iminds.iot.dianne.nn.train.strategy;

import java.util.List;

import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.tensor.Tensor;

public class StochasticGradient implements Trainer {

	@Override
	public void train(Input input, Output output, List<Trainable> modules, 
			Criterion criterion, Dataset data) {
		System.out.println("Starting training");
		
		// Training procedure
		int batchSize = 10;
		int noEpochs = 10;
		
		for(int epoch=0;epoch<noEpochs;epoch++){
			
			System.out.println(epoch);
			
			int batch = 0;
			for(int i=0;i<data.size();i++){

				// Read samples from dataset
				Tensor in = data.getInputSample(i);
				
				// Forward through input module
				input.forward(input.getId(), in);
				
//				// TODO output callback/sync execution flag?
//				try {
//					Thread.sleep(100);
//				} catch (InterruptedException e) {}
				
				Tensor mse = criterion.forward(output.getOutput(), data.getOutputSample(i));
				
				// Backward through output module
				output.backward(output.getId(), criterion.backward(output.getOutput(), data.getOutputSample(i)));
			
//				// TODO input callback/sync execution flag?
//				try {
//					Thread.sleep(100);
//				} catch (InterruptedException e) {}
				
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
		}
		// repeat
			
			
	}

}
