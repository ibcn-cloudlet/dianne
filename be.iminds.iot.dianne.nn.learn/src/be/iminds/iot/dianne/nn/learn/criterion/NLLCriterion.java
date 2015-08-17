package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class NLLCriterion implements Criterion {

	protected final TensorFactory factory;
	
	protected Tensor gradInput;
	protected Tensor nll;
	
	public NLLCriterion(TensorFactory factory) {
		this.factory = factory;	
		this.nll = factory.createTensor(1);
	}
	
	@Override
	public Tensor error(final Tensor output, final Tensor target) {
		float ll = factory.getTensorMath().dot(factory.getTensorMath().log(null, output), target);
		nll.set(-ll, 0);
		return nll;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		gradInput = factory.getTensorMath().cdiv(gradInput, target, output);
		gradInput = factory.getTensorMath().mul(gradInput, gradInput, -1.0f);
		return gradInput;
	}
}
