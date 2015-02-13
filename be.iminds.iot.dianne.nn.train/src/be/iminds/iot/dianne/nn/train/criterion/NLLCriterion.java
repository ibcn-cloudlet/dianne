package be.iminds.iot.dianne.nn.train.criterion;

import be.iminds.iot.dianne.nn.train.Criterion;
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
	public Tensor forward(final Tensor output, final Tensor target) {
		nll.set(-output.get(factory.getTensorMath().argmax(target)), 0);
		return nll;
	}

	@Override
	public Tensor backward(final Tensor output, final Tensor target) {
		if(gradInput == null){
			gradInput = factory.createTensor(target.dims());
		}
		gradInput.fill(0.0f);
		gradInput.set(-1.0f, factory.getTensorMath().argmax(target));
		
		return gradInput;
	}
}
