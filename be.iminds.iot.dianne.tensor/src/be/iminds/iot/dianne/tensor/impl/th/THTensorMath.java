package be.iminds.iot.dianne.tensor.impl.th;

import be.iminds.iot.dianne.tensor.TensorMath;

public class THTensorMath implements TensorMath<THTensor> {

	@Override
	public THTensor add(THTensor res, THTensor tensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor add(THTensor res, THTensor tensor1, THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor add(THTensor res, THTensor tensor1, float value,
			THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor sub(THTensor res, THTensor tensor, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor sub(THTensor res, THTensor tensor1, THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor sub(THTensor res, THTensor tensor1, float value,
			THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor mul(THTensor res, THTensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor cmul(THTensor res, THTensor tensor1, THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor div(THTensor res, THTensor tensor1, float value) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor cdiv(THTensor res, THTensor tensor1, THTensor tensor2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float dot(THTensor vec1, THTensor vec2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public THTensor vv(THTensor res, THTensor vec1, THTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor mv(THTensor res, THTensor mat, THTensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor tmv(THTensor res, THTensor mat, THTensor vec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor mm(THTensor res, THTensor mat1, THTensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor addvv(THTensor res, THTensor mat, THTensor vec1,
			THTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor addmv(THTensor res, THTensor vec1, THTensor mat,
			THTensor vec2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor addmm(THTensor res, THTensor mat, THTensor mat1,
			THTensor mat2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor exp(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor log(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor tanh(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor dtanh(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor sigmoid(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor dsigmoid(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor thresh(THTensor res, THTensor tensor, float thresh,
			float coeff, float offset) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor thresh(THTensor res, THTensor tensor, THTensor threshs,
			THTensor coeffs, THTensor offsets) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor dthresh(THTensor res, THTensor tensor, float thresh,
			float coeff) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor dthresh(THTensor res, THTensor tensor, THTensor threshs,
			THTensor coeffs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor softmax(THTensor res, THTensor tensor) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float sum(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float max(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float min(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float mean(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int argmax(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int argmin(THTensor tensor) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public THTensor convolution2D(THTensor res, THTensor mat1, THTensor mat2,
			int sx, int sy, int mode, boolean flip) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor addconvolution2D(THTensor res, THTensor mat, THTensor mat1,
			THTensor mat2, int sx, int sy, int mode, boolean flip) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor maxpool2D(THTensor res, THTensor mat, int w, int h, int sx,
			int sy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor dmaxpool2D(THTensor res, THTensor mat2, THTensor mat1,
			int w, int h, int sx, int sy) {
		// TODO Auto-generated method stub
		return null;
	}


}
