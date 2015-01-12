package be.iminds.iot.dianne.nn.train;

import java.util.List;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Trainable;

public interface Trainer {

	// TODO better specify the neural network to train?
	public void train(Module input, Module output, List<Trainable> module, Dataset data);
}
