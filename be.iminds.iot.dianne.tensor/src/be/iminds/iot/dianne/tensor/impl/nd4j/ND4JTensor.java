package be.iminds.iot.dianne.tensor.impl.nd4j;

import be.iminds.iot.dianne.tensor.Tensor;

public class ND4JTensor implements Tensor<ND4JTensor> {

	@Override
	public int dim() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int[] dims() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int size(int d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void reshape(int... d) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public float get(int... d) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public float[] get() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void set(float v, int... d) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void set(float[] data) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void fill(float v) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void rand() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void randn() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean sameDim(Tensor<?> other) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean hasDim(int... dims) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public ND4JTensor copyInto(ND4JTensor other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor narrow(int dim, int index, int size) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor narrow(int... ranges) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor select(int dim, int index) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor transpose(ND4JTensor res, int d1, int d2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ND4JTensor diag(ND4JTensor res) {
		// TODO Auto-generated method stub
		return null;
	}

}
