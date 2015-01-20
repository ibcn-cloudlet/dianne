package be.iminds.iot.dianne.nn.train;

import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;

public interface Evaluator {

	// TODO better specify the neural network to train?
	public Evaluation evaluate(final Input input, final Output output, final Dataset data);

}
