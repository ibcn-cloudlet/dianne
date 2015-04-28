package be.iminds.iot.dianne.tensor.impl.th;

import be.iminds.iot.dianne.tensor.Tensor;

public class THTensor implements Tensor<THTensor> {

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
	public THTensor copyInto(THTensor other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor narrow(int dim, int index, int size) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor narrow(int... ranges) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor select(int dim, int index) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor transpose(THTensor res, int d1, int d2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor diag(THTensor res) {
		// TODO Auto-generated method stub
		return null;
	}


}
