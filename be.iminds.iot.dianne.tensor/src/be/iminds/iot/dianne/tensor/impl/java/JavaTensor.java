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
				s+="[";
				for(int j = 0; j < dims[1]-1; j++){
					s+= get(i,j)+", ";
				}
				s+= get(i, dims[1]-1) + "]";
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
	public boolean sameDim(Tensor<?> other) {
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

	@Override
	public JavaTensor clone(JavaTensor other) {
		if(other == null
				|| this.data.length != other.data.length)
			other = new JavaTensor(dims);
		
		System.arraycopy(data, 0, other.data, 0, data.length);
		
		return other;
	}

	@Override
	public JavaTensor transpose(JavaTensor res, int d1, int d2) {
		res = this.clone(res);
		
		if(res.dims.length <= d1 || res.dims.length <= d2){
			int maxd = d1 < d2 ? d2 : d1;
			
			int[] newDims = new int[maxd+1];
			int[] newStrides = new int[maxd+1];
			
			System.arraycopy(res.dims, 0, newDims, 0, res.dims.length);
			System.arraycopy(res.strides, 0, newStrides, 0, res.strides.length);
			
			for(int i = res.dims.length; i <= maxd; i++){
				newDims[i] = 1;
				newStrides[i] = 1;
			}
			
			res.dims = newDims;
			res.strides = newStrides;
		}
		
		int tempDim = res.dims[d1];
		res.dims[d1] = res.dims[d2];
		res.dims[d2] = tempDim;
		
		int tempStride = res.strides[d1];
		res.strides[d1] = res.strides[d2];
		res.strides[d2] = tempStride;
		
		return res;
	}
}
