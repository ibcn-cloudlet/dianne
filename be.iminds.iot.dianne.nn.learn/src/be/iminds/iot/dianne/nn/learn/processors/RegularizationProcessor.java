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
	
	public RegularizationProcessor( AbstractProcessor p) {
		super(p.factory, p.input, p.output, p.toTrain, p.dataset, p.config, p.logger);
		decorated = p;
		
		String r = config.get("regularization");
		if(r!=null){
			regularization = Float.parseFloat(r);
		}
		
		System.out.println("Regularization");
		System.out.println("* factor = "+regularization);
		System.out.println("---");
	}
	
	@Override
	public float processNext() {
		float error = decorated.processNext();
		
		// subtract previous parameters
		toTrain.entrySet().stream().forEach(e -> {
			Tensor params = e.getValue().getParameters();
			Tensor deltaParams = e.getValue().getDeltaParameters();
			factory.getTensorMath().sub(deltaParams, deltaParams, regularization, params);
		});
		
		return error;
	}

}
