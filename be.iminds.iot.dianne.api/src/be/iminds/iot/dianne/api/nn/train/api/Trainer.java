package be.iminds.iot.dianne.api.nn.train.api;

import java.util.List;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;

public interface Trainer {

	public void train(final Input input, final Output output, 
			final List<Trainable> module, final List<Preprocessor> preprocessors, 
			final Criterion criterion, final Dataset data);
	
}
