package be.iminds.iot.dianne.nn.train;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;

public interface Evaluator {

	public Evaluation evaluate(final Input input, final Output output, final Dataset data);

}
