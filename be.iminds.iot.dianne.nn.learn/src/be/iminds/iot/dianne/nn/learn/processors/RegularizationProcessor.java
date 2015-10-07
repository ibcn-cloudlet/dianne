package be.iminds.iot.dianne.nn.learn.processors;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Additional learning techniques like Momentum can be implemented as a Processor decorator
 */
public class RegularizationProcessor extends AbstractProcessor {

	private final Processor decorated;
	
	private float regularization = 0.001f;
	
	private Map<UUID, Tensor> previousDelta = new HashMap<UUID, Tensor>();
	
	public RegularizationProcessor( AbstractProcessor p, float regularization) {
		super(p.factory, p.nn, p.logger);
		this.decorated = p;

		this.regularization = regularization;
	}
	
	@Override
	public float processNext() {
		float error = decorated.processNext();
		
		// subtract previous parameters
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor params = e.getValue().getParameters();
			Tensor deltaParams = e.getValue().getDeltaParameters();
			factory.getTensorMath().sub(deltaParams, deltaParams, regularization, params);
		});
		
		return error;
	}

}
