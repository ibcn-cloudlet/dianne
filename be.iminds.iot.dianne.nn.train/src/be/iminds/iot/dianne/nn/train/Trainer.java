package be.iminds.iot.dianne.nn.train;

import java.util.List;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Module;

public interface Trainer {

	public void train(final List<Module> modules, 
			final Criterion criterion, final Dataset data);
	
}
