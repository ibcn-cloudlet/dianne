package be.iminds.iot.dianne.nn.train.criterion;

import be.iminds.iot.dianne.api.nn.train.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MSECriterion implements Criterion {

	protected final TensorFactory factory;
		
	protected Tensor error;
	protected Tensor sqerror;
	protected Tensor mse;
	
	public MSECriterion(TensorFactory factory) {
		this.factory = factory;	
		this.mse = factory.createTensor(1);
	}
	
	@Override
	public Tensor forward(final Tensor output, final Tensor target) {
		error = factory.getTensorMath().sub(error, output, target);
		sqerror = factory.getTensorMath().cmul(sqerror, error, error);
		mse.set(factory.getTensorMath().sum(sqerror)*0.5f, 0);
		return mse;
	}

	@Override
	public Tensor backward(final Tensor output, final Tensor target) {
		return error;
	}

}
