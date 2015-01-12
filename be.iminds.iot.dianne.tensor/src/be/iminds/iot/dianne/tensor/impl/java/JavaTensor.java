package be.iminds.iot.dianne.tensor.impl.java;

import java.util.Arrays;
import java.util.Random;

import be.iminds.iot.dianne.tensor.Tensor;

public class JavaTensor implements Tensor<JavaTensor> {

	int[] dims;
	int offset = 0;
	int[] strides;
	float[] data;
	
	public JavaTensor(final int... d){
		this(d, null);
	
		data = new float[size()];
	}

	JavaTensor(final int[] dims, final float[] data){
		this.dims = dims;
		this.data = data;

		this.strides = new int[dims.length];
		int stride = 1;
		for(int i=dims.length-1;i>=0;i--){
			strides[i] = stride;
			stride*= dims[i];
		}
		
		if(data!=null){
			assert data.length == size();
		}
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
		int size = 1;
		for(int i=0;i<dims.length;i++){
			size *= dims[i];
		}
		return size;
	}

	@Override
	public int size(int d) {
		return dims[d];
	}
	
	@Override
	public float get(final int... d) {
		assert d.length == dims.length;
		
		int index = getIndex(d);
		return data[index];
	}
	
	@Override
	public void set(final float v, final int... d) {
		assert d.length == dims.length;
		
		int index = getIndex(d);
		data[index] = v;
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
		JavaTensorIterator it = iterator();
		while(it.hasNext()){
			data[it.next()] = v;
		}
	}

	@Override
	public void rand() {
		Random r = new Random(System.currentTimeMillis());
		JavaTensorIterator it = iterator();
		while(it.hasNext()){
			data[it.next()] = r.nextFloat();
		}
	}

	@Override
	public boolean equals(Object other){
		if(!(other instanceof JavaTensor)){
			return false;
		} 
		JavaTensor o = (JavaTensor) other;
		
		if(o.dim() != dim()){
			return false;
		}
		
		if(o.size() != size()){
			return false;
		}
		
		for(int i=0;i<dims.length;i++){
			if(o.size(i) != dims[i]){
				return false;
			}
		}
		
		JavaTensorIterator it1 = iterator();
		JavaTensorIterator it2 = o.iterator();
		while(it1.hasNext()){
			if(data[it1.next()] != o.data[it2.next()])
				return false;
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
		
		JavaTensorIterator it = iterator();
		int i = 0;
		while(it.hasNext()){
			other.data[i++] = data[it.next()];
		}
		
		return other;
	}

	@Override
	public JavaTensor narrow(int dim, int index, int size) {
		int[] narrowDims = dims.clone();
		JavaTensor narrow = new JavaTensor(narrowDims, data);
		narrow.dims[dim] = size;
		int[] start = new int[dims.length];
		start[dim] = index;
		narrow.offset = getIndex(start);
		return narrow;
	}

	@Override
	public JavaTensor narrow(int... ranges) {
		int[] narrowDims = dims.clone();
		JavaTensor narrow = new JavaTensor(narrowDims, data);
		int[] start = new int[dims.length];
		for(int i=0;i<ranges.length/2;i++){
			start[i] = ranges[2*i];
			narrow.dims[i] = ranges[2*i+1]-ranges[2*i]+1;
		}
		narrow.offset = getIndex(start);
		return narrow;
	}

	@Override
	public JavaTensor select(int dim, int index) {
		return narrow(dim, index, 1);
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
	
	int getIndex(int[] d){
		int index = offset;
		for(int i=0;i<d.length;i++){
			index += strides[i]*d[i];
		}
		return index;
	}
	
	JavaTensorIterator iterator(){
		return new JavaTensorIterator();
	}
	
	class JavaTensorIterator {
		private int[] index = new int[dims.length]; // 3D index
		private int current = 0;
		private int next = offset; // linear index
		
		public int next(){
			current = next;
			boolean incremented = false;
			int dim = dims.length-1;
			while(!incremented){
				index[dim]++;
				if(index[dim]==dims[dim]){
					index[dim] = 0;
					dim--;
				} else {
					incremented = true;
					//next+= strides[dim];
					next = getIndex(index);
				}
				if(dim<0){
					next = -1;
					incremented = true; 
				}
			}
			return current;
		}
		
		public boolean hasNext(){
			return next != -1;
		}
	}
}
