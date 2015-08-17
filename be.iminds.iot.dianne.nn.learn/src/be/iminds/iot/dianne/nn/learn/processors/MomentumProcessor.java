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
	
	private Map<UUID, Tensor> previousGrad = new HashMap<UUID, Tensor>();
	
	public MomentumProcessor( AbstractProcessor p,
			Map<String, String> config) {
		super(p.factory, p.input, p.output, p.toTrain, p.dataset);
		decorated = p;
		
		// TODO set momentum based on config
		momentum = 0.001f;
	}
	
	@Override
	public float processNext() {
		float error = decorated.processNext();
		
		// add momentum
		toTrain.entrySet().stream().forEach(e -> {
			Tensor prev = previousGrad.get(e.getKey());
			if(prev!=null){
				Tensor gradParams = e.getValue().getGradParameters();
				factory.getTensorMath().add(gradParams, gradParams , momentum, prev);
			}
		});
		
		// copy to previousGrad
		toTrain.entrySet().stream().forEach(e -> {
			Tensor prev = previousGrad.get(e.getKey());
			Tensor gradParams = e.getValue().getGradParameters();
			gradParams.copyInto(prev);
			previousGrad.put(e.getKey(), gradParams);
		});
		
		return error;
	}

}
