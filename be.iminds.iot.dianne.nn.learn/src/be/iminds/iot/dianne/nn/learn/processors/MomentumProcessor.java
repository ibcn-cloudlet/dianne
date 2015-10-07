package be.iminds.iot.dianne.nn.learn.processors;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Additional learning techniques like Momentum can be implemented as a Processor decorator
 */
public class MomentumProcessor extends AbstractProcessor {

	private final Processor decorated;
	private final float momentum;
	
	private Map<UUID, Tensor> previousDelta = new HashMap<UUID, Tensor>();
	
	public MomentumProcessor( AbstractProcessor p, float momentum ) {
		super(p.factory, p.nn, p.logger);
		this.decorated = p;
		
		this.momentum = momentum;
	}
	
	@Override
	public float processNext() {
		float error = decorated.processNext();
		
		// add momentum
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor prev = previousDelta.get(e.getKey());
			if(prev!=null){
				Tensor deltaParams = e.getValue().getDeltaParameters();
				factory.getTensorMath().add(deltaParams, deltaParams , momentum, prev);
			}
		});
		
		// copy to previousGrad
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor prev = previousDelta.get(e.getKey());
			Tensor deltaParams = e.getValue().getDeltaParameters();
			deltaParams.copyInto(prev);
			previousDelta.put(e.getKey(), deltaParams);
		});
		
		return error;
	}

}
