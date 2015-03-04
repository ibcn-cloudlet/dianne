package be.iminds.iot.dianne.api.nn.train.api;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;

public interface Evaluator {

	public Evaluation evaluate(final Input input, final Output output, final Dataset data);

}
