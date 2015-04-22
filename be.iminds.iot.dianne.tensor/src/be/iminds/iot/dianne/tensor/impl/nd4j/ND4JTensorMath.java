package be.iminds.iot.dianne.tensor.impl.nd4j;

import be.iminds.iot.dianne.tensor.TensorMath;

public class ND4JTensorMath implements TensorMath<ND4JTensor> {

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor1, ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor add(ND4JTensor res, ND4JTensor tensor1, float value,
			ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor1, ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor sub(ND4JTensor res, ND4JTensor tensor1, float value,
			ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor mul(ND4JTensor res, ND4JTensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor cmul(ND4JTensor res, ND4JTensor tensor1,
			ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor div(ND4JTensor res, ND4JTensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor div(ND4JTensor res, ND4JTensor tensor1, ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor cdiv(ND4JTensor res, ND4JTensor tensor1,
			ND4JTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float dot(ND4JTensor vec1, ND4JTensor vec2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public ND4JTensor vv(ND4JTensor res, ND4JTensor vec1, ND4JTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor mv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor tmv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor mm(ND4JTensor res, ND4JTensor mat1, ND4JTensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor addvv(ND4JTensor res, ND4JTensor mat, ND4JTensor vec1,
			ND4JTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor addmv(ND4JTensor res, ND4JTensor vec1, ND4JTensor mat,
			ND4JTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor addmm(ND4JTensor res, ND4JTensor mat, ND4JTensor mat1,
			ND4JTensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor exp(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor log(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor tanh(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dtanh(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor sigmoid(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dsigmoid(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor thresh(ND4JTensor res, ND4JTensor tensor, float thresh,
			float coeff, float offset) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor thresh(ND4JTensor res, ND4JTensor tensor,
			ND4JTensor threshs, ND4JTensor coeffs, ND4JTensor offsets) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dthresh(ND4JTensor res, ND4JTensor tensor, float thresh,
			float coeff) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dthresh(ND4JTensor res, ND4JTensor tensor,
			ND4JTensor threshs, ND4JTensor coeffs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor softmax(ND4JTensor res, ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float sum(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float max(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float min(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float mean(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int argmax(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int argmin(ND4JTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public ND4JTensor convolution2D(ND4JTensor res, ND4JTensor mat1,
			ND4JTensor mat2, int sx, int sy, int mode, boolean flip) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor maxpool2D(ND4JTensor res, ND4JTensor mat, int w, int h,
			int sx, int sy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor dmaxpool2D(ND4JTensor res, ND4JTensor mat2,
			ND4JTensor mat1, int w, int h, int sx, int sy) {
		// TODO Auto-generated method stub
		return null;
	}

}
