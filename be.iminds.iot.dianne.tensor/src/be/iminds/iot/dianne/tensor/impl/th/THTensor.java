package be.iminds.iot.dianne.tensor.impl.th;

import java.nio.ByteBuffer;
import java.util.Arrays;

import be.iminds.iot.dianne.tensor.Tensor;

public class THTensor implements Tensor<THTensor> {

	long address;
	int[] dims;
	
	public THTensor(int[] dims) {
		this(null, dims);
	}
	
	public THTensor(float[] data, int[] dims) {
		this.dims = dims;
		this.address = init(data, dims);
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
	public int size(final int d) {
		return dims[d];
	}
	
	@Override
	public void reshape(int... d) {
		this.dims = d;
		
	}

	@Override
	public float get(int... d) {
		return get(address, d);
	}

	@Override
	public float[] get() {
		ByteBuffer b = get(address);
		return b.asFloatBuffer().array();
	}

	@Override
	public void set(float v, int... d) {
		set(address, v, d);
	}

	@Override
	public void set(float[] data) {
		set(address, data);
	}

	@Override
	public void fill(float v) {
		fill(address, v);
	}

	@Override
	public void rand() {
		rand(address);
	}

	@Override
	public void randn() {
		randn(address);
	}

	@Override
	public boolean sameDim(final Tensor<?> other) {
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
	public boolean hasDim(final int... dims){
		if(this.dims.length!=dims.length){
			return false;
		}
		for(int i=0;i<dims.length;i++){
			if(this.dims[i] != dims[i]){
				return false;
			}
		}
		return true;
	}

	@Override
	public THTensor copyInto(THTensor other) {
		copyInto(address, other.address);
		return other;
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

	@Override
	public boolean equals(Object other){
		if(!(other instanceof THTensor)){
			return false;
		} 
		THTensor o = (THTensor) other;
		return equals(address, o.address);
	}
	
	@Override
	public int hashCode(){
		return (int)address;
	}
	
	public void finalize(){
		free(address);
	}
	
	private native long init(float[] data, int[] dims);
	
	private native void free(long address);
	
	private native void reshape(long src, int... d);

	private native float get(long src, int... d);

	private native ByteBuffer get(long src);
	
	private native void set(long src, float v, int... d);

	private native void set(long src, float[] data);
	
	private native void fill(long src, float v);

	private native void rand(long src);

	private native void randn(long src);

	private native long copyInto(long src, long other);

	private native long narrow(long src, int dim, int index, int size);

	private native long select(long src, int dim, int index);

	private native long transpose(long src, long res, int d1, int d2);

	private native long diag(long src, long res);
	
	private native boolean equals(long src, long other);
}
