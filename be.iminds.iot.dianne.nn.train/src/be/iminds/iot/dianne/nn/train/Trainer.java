package be.iminds.iot.dianne.nn.train;

import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;

public interface Trainer {

	public void train(final Input input, final Output output, 
			final List<Trainable> module, final List<Preprocessor> preprocessors, 
			final Criterion criterion, final Dataset data);
	
}
