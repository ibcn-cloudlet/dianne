package be.iminds.iot.dianne.nn.train.strategy;

import java.util.List;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Dataset;
import be.iminds.iot.dianne.nn.train.Trainer;

public class StochasticGradient implements Trainer {

	@Override
	public void train(Module input, Module output, List<Trainable> module,
			Dataset data) {
		// Training procedure
		
		// Read samples from dataset
		
		// Forward through input module
		
		// Backward through output module
		
		// accGradParameters for all trainable modules
		
		// updateParameters after batch
		
		// repeat
	}

}
