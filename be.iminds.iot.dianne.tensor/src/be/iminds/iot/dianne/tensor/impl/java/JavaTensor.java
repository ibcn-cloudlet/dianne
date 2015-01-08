package be.iminds.iot.dianne.tensor.impl.java;

import java.util.Arrays;

import be.iminds.iot.dianne.tensor.Tensor;

public class JavaTensor implements Tensor<JavaTensor> {

	int[] dims;
	int[] strides;
	float[] data;
	
	JavaTensor(final int... d){
		this(d, null);
		
		int size = 1;
		for(int i=0;i<d.length;i++){
			size *= d[i];
		}
		data = new float[size];
	}

	JavaTensor(final int[] dims,final float[] data){
		this.dims = dims;
		this.data = data;

		this.strides = new int[dims.length];
		int stride = 1;
		for(int i=dims.length-1;i>=0;i--){
			strides[i] = stride;
			stride*= dims[i];
		}
		
		// TODO check data size?
	}
	
	@Override
	public int dim() {
		return dims.length;
	}

	@Override
	public int size() {
		return data.length;
	}

	@Override
	public int size(int d) {
		return dims[d];
	}
	
	@Override
	public float get(final int... d) {
		// TODO check indices?
		int index = 0;
		for(int i=0;i<d.length;i++){
			index += strides[i]*d[i];
		}
		return data[index];
	}

	@Override
	public void set(final float v, final int... d) {
		// TODO check indices?
		int index = 0;
		for(int i=0;i<d.length;i++){
			index += strides[i]*d[i];
		}
		data[index] = v;
	}

	@Override
	public String toString(){
		if(dims.length == 2){
			// format as matrix
			String s = "";
			for(int i=0;i<dims[0];i++){
				s+=Arrays.toString(Arrays.copyOfRange(data, strides[0]*i, strides[0]*i+dims[1]));
				s+="\n";
			}
			return s;
		}
		return Arrays.toString(data);
	}

}
