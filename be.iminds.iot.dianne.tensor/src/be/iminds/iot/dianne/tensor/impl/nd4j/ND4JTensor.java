package be.iminds.iot.dianne.tensor.impl.nd4j;

import java.util.Arrays;
import java.util.Random;

import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import be.iminds.iot.dianne.tensor.Tensor;

public class ND4JTensor implements Tensor<ND4JTensor> {

	static Random random = new Random(1234);
	
	INDArray data;
	
	ND4JTensor(INDArray d) {
		this.data = d;
	}
	
	@Override
	public int dim() {
		return data.shape().length;
	}

	@Override
	public int[] dims() {
		return Arrays.copyOf(data.shape(), data.shape().length);
	}

	@Override
	public int size() {
		return data.length();
	}

	@Override
	public int size(int d) {
		return data.size(d);
	}

	@Override
	public void reshape(int... d) {
		data = data.reshape(d);
	}

	@Override
	public float get(int... d) {
		return data.getFloat(d);
	}

	@Override
	public float[] get() {
		return data.data().asFloat();
	}

	@Override
	public void set(float v, int... d) {
		data.put(d, Nd4j.create(new float[]{v}));
	}

	@Override
	public void set(float[] data) {
		this.data.data().assign(new FloatBuffer(data));
	}

	@Override
	public void fill(float v) {
		int[] dims = data.shape();
		int[] index = new int[dims.length];
		boolean next = true;
		while(next){
			boolean incremented = false;
			int dim = dims.length-1;
			while(!incremented){
				index[dim]++;
				if(index[dim]==dims[dim]){
					index[dim] = 0;
					dim--;
				} else {
					incremented = true;
				}
				if(dim<0){
					next = false;
					incremented = true; 
				}
			}
			data.put(index, Nd4j.create(new float[]{v}));
		}
	
		// TODO this should be done using an operation
		//Nd4j.getExecutioner().exec(new ScalarAdd(data, v));
		// .. or the assign function?
		//data.assign(v);
	}

	@Override
	public void rand() {
		int[] dims = data.shape();
		int[] index = new int[dims.length];
		boolean next = true;
		while(next){
			boolean incremented = false;
			int dim = dims.length-1;
			while(!incremented){
				index[dim]++;
				if(index[dim]==dims[dim]){
					index[dim] = 0;
					dim--;
				} else {
					incremented = true;
				}
				if(dim<0){
					next = false;
					incremented = true; 
				}
			}
			data.put(index, Nd4j.create(new float[]{random.nextFloat()}));
		}
	}

	@Override
	public void randn() {
		int[] dims = data.shape();
		int[] index = new int[dims.length];
		boolean next = true;
		while(next){
			boolean incremented = false;
			int dim = dims.length-1;
			while(!incremented){
				index[dim]++;
				if(index[dim]==dims[dim]){
					index[dim] = 0;
					dim--;
				} else {
					incremented = true;
				}
				if(dim<0){
					next = false;
					incremented = true; 
				}
			}
			data.put(index, Nd4j.create(new float[]{(float)random.nextGaussian()}));
		}
	}

	@Override
	public boolean sameDim(Tensor<?> other) {
		int[] dims = data.shape();
		if(dims.length!=other.dim()){
			return false;
		}
		for(int i=0;i<data.shape().length;i++){
			if(other.size(i) != dims[i]){
				return false;
			}
		}
		return true;
	}

	@Override
	public boolean hasDim(int... d) {
		int[] dims = data.shape();
		if(dims.length!=d.length){
			return false;
		}
		for(int i=0;i<d.length;i++){
			if(dims[i] != d[i]){
				return false;
			}
		}
		return true;
	}

	@Override
	public ND4JTensor copyInto(ND4JTensor other) {
		if(other==null){
			INDArray dup = data.dup();
			return new ND4JTensor(dup);
		} else {
			Nd4j.getBlasWrapper().copy(this.data.linearView(), other.data.linearView());
			return other;
		}
	}

	@Override
	public ND4JTensor narrow(int dim, int index, int size) {
		int[] dims = dims();
		// set offset
		int[] offsets = new int[dims.length];
		offsets[dim] = index;
		// set strides
		int[] strides = new int[dims.length];
		for(int i=strides.length-1;i>=0;i--){
			strides[i] = (i+1>=strides.length) ? 1 : strides[i+1]*dims[i+1];
		}
		// crop dim
		dims[dim] = size;
		
		return new ND4JTensor(data.subArray(offsets, dims, strides));
	}

	@Override
	public ND4JTensor narrow(int... ranges) {
		int[] dims = dims();
		// set offset
		int[] offsets = new int[dims.length];
		for(int i=0;i<ranges.length/2;i++){
			offsets[i] = ranges[i*2];
		}
		// set strides
		int[] strides = new int[dims.length];
		for(int i=strides.length-1;i>=0;i--){
			strides[i] = (i+1>=strides.length) ? 1 : strides[i+1]*dims[i+1];
		}
		// crop dims
		for(int i=0;i<ranges.length/2;i++){
			dims[i] = ranges[i*2+1];
		}
		
		return new ND4JTensor(data.subArray(offsets, dims, strides));
	}

	@Override
	public ND4JTensor select(int dim, int index) {
		INDArray slice = data.slice(index, dim);
		int[] oldshape = dims();
		int[] newshape = new int[oldshape.length-1];
		int k = 0;
		for(int i=0;i<oldshape.length;i++){
			if(k==dim){
				i+=1;
			}
			newshape[k++] = oldshape[i];
		}
		return new ND4JTensor(slice.reshape(newshape));
	}

	@Override
	public ND4JTensor transpose(ND4JTensor res, int d1, int d2) {
		INDArray transposed = data.swapAxes(d1, d2);
		if(res==null){
			return new ND4JTensor(transposed);
		} else {
			Nd4j.getBlasWrapper().copy(transposed.linearView(), res.data.linearView());
			return res;
		}
	}

	@Override
	public ND4JTensor diag(ND4JTensor res) {
		// only works for 2 dim nxn matrix
		int[] offsets = new int[]{0};
		int[] dims = new int[]{data.size(0)};
		int[] stride = new int[]{data.size(0)+1};
		
		INDArray diag = data.linearView().subArray(offsets, dims, stride);
		
		if(res==null){
			return new ND4JTensor(diag);
		} else {
			res.data = diag;
			return res;
		}
	}

	@Override
	public boolean equals(Object other){
		if(!(other instanceof ND4JTensor)){
			return false;
		} 
		ND4JTensor o = (ND4JTensor) other;
		
		return data.equals(o.data);
	}
	
	@Override
	public String toString(){
		return data.toString();
	}
}
