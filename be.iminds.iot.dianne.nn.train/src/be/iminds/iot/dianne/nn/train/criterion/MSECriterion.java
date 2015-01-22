package be.iminds.iot.dianne.nn.train.criterion;

import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class MSECriterion implements Criterion {

	// the factory for this module
	protected static final TensorFactory factory = new JavaTensorFactory();
		
	protected Tensor error;
	protected Tensor sqerror;
	protected Tensor mse = factory.createTensor(1);
	
	@Override
	public Tensor forward(final Tensor output, final Tensor target) {
		error = factory.getTensorMath().sub(error, target, output);
		sqerror = factory.getTensorMath().cmul(sqerror, error, error);
		mse.set(factory.getTensorMath().sum(sqerror)*0.5f, 0);
		return mse;
	}

	@Override
	public Tensor backward(final Tensor output, final Tensor target) {
		return error;
	}

}
