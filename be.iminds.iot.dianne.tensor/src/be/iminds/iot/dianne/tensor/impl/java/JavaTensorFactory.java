package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorFactory implements TensorFactory {

	private final JavaTensorMath math = new JavaTensorMath(this);
	
	@Override
	public Tensor createTensor(final int... d) {
		return new JavaTensor(d);
	}

	@Override
	public TensorMath getTensorMath() {
		return math;
	}

}
