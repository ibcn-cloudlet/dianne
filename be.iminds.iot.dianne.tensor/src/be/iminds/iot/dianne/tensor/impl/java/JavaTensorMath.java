package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath {

	private final JavaTensorFactory factory;
	
	JavaTensorMath(JavaTensorFactory factory) {
		this.factory = factory;
	}

	@Override
	public Tensor add(Tensor res, Tensor tensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor add(Tensor res, Tensor tensor1, Tensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor add(Tensor res, Tensor tensor1, float value, Tensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor mul(Tensor res, Tensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor cmul(Tensor res, Tensor tensor1, Tensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor div(Tensor res, Tensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor cdiv(Tensor res, Tensor tensor1, Tensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float dot(Tensor vec1, Tensor vec2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Tensor mv(Tensor res, Tensor mat, Tensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor mm(Tensor res, Tensor mat1, Tensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}

}
