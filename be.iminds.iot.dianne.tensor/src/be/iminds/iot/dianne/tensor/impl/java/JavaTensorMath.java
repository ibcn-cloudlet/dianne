package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath<JavaTensor> {

	private final JavaTensorFactory factory;
	
	JavaTensorMath(JavaTensorFactory factory) {
		this.factory = factory;
	}

	@Override
	public JavaTensor add(JavaTensor res, JavaTensor JavaTensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor add(JavaTensor res, JavaTensor JavaTensor1, JavaTensor JavaTensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor add(JavaTensor res, JavaTensor JavaTensor1, float value, JavaTensor JavaTensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor mul(JavaTensor res, JavaTensor JavaTensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor cmul(JavaTensor res, JavaTensor JavaTensor1, JavaTensor JavaTensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor div(JavaTensor res, JavaTensor JavaTensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor cdiv(JavaTensor res, JavaTensor JavaTensor1, JavaTensor JavaTensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float dot(JavaTensor vec1, JavaTensor vec2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public JavaTensor mv(JavaTensor res, JavaTensor mat, JavaTensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public JavaTensor mm(JavaTensor res, JavaTensor mat1, JavaTensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}
}
