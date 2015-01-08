package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorFactory implements TensorFactory<JavaTensor> {

	private final TensorMath<JavaTensor> math = new JavaTensorMath();
	
	@Override
	public JavaTensor createTensor(final int... d) {
		return new JavaTensor(d);
	}

	@Override
	public TensorMath<JavaTensor> getTensorMath() {
		return math;
	}

}
