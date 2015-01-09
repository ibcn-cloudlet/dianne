package be.iminds.iot.dianne.tensor.impl.java;

import java.util.Arrays;
import java.util.Random;

import be.iminds.iot.dianne.tensor.Tensor;

public class JavaTensor implements Tensor<JavaTensor> {

	int[] dims;
	int[] strides;
	float[] data;
	
	public JavaTensor(final int... d){
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
	public int[] dims() {
		return Arrays.copyOf(dims, dims.length);
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
	public float get(final int i) {
		// TODO check indices?
		return data[i];
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
	public void set(final float v, final int i) {
		// TODO check indices?
		data[i] = v;
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

	@Override
	public void fill(float v) {
		for(int i=0;i<data.length;i++){
			data[i] = v;
		}
	}

	@Override
	public void rand() {
		Random r = new Random(System.currentTimeMillis());
		for(int i=0;i<data.length;i++){
			data[i] = r.nextFloat();
		}
	}

	@Override
	public boolean equals(Object other){
		if(!(other instanceof Tensor)){
			return false;
		} 
		Tensor<?> o = (Tensor<?>) other;
		
		if(o.dim() != dims.length){
			return false;
		}
		
		if(o.size() != data.length){
			return false;
		}
		
		for(int i=0;i<dims.length;i++){
			if(o.size(i) != dims[i]){
				return false;
			}
		}
		
		for(int i=0;i<data.length;i++){
			if(o.get(i) != data[i]){
				return false;
			}
		}
		
		return true;
		
	}

	@Override
	public boolean sameDim(Tensor other) {
		if(dims.length!=other.dim()){
			return false;
		}
		for(int i=0;i<dims.length;i++){
			if(other.size(i) != dims[i]){
				return false;
			}
		}
		return true;
	}
}
