/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.tensor.impl.th;

import java.util.Arrays;

import be.iminds.iot.dianne.tensor.Tensor;

public class THTensor implements Tensor<THTensor> {

	long address;
	int[] dims;
	
	THTensor(int[] dims) {
		this(null, dims);
	}
	
	THTensor(float[] data, int[] dims) {
		this.dims = dims;
		this.address = init(data, dims);
	}
	
	THTensor(long address) {
		this.address = address;
		this.dims = dims(address);
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
		reshape(address, d);
	}

	@Override
	public float get(int... d) {
		return get(address, d);
	}

	@Override
	public float[] get() {
		return get(address);
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
	public void bernoulli(float p) {
		bernoulli(address, p);
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
		THTensor copy = other;
		if(copy==null){
			copy = new THTensor(dims);
		}
		copyInto(address, copy.address);
		return copy;
	}

	@Override
	public THTensor narrow(int dim, int index, int size) {
		long n = narrow(address, dim, index, size);
		return new THTensor(n);
	}

	@Override
	public THTensor narrow(int... ranges) {
		THTensor n = this;
		for(int i=0;i<ranges.length-1;i+=2){
			n = n.narrow(i/2, ranges[i], ranges[i+1]); 
		}
		return n;
	}

	@Override
	public THTensor select(int dim, int index) {
		long s = select(address, dim, index);
		return new THTensor(s);
	}

	@Override
	public THTensor transpose(THTensor res, int d1, int d2) {
		long l = transpose(address, d1, d2);
		// TH does not do transpose to res vector, just copy here
		THTensor t = new THTensor(l);
		if(res!=null){
			t.copyInto(res);
			return res;
		} else {
			return t;
		}
	}

	@Override
	public THTensor diag(THTensor res) {
		long l = diag(address, res == null ?  0 : res.address);
		return res==null ? new THTensor(l) : res;
	}

	@Override
	public boolean equals(Object other){
		if(!(other instanceof THTensor)){
			return false;
		} 
		THTensor o = (THTensor) other;
		return equals(o);
	}
	
	@Override
	public boolean equals(THTensor other){
		return equals(other, 0.0f);
	}
	
	@Override
	public boolean equals(THTensor other, float threshold){
		if(!this.sameDim(other)){
			return false;
		}
		return equals(address, other.address, threshold);
	}
	
	@Override
	public int hashCode(){
		return (int)address;
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append(Arrays.toString(dims));
		b.append("\n");
		b.append(Arrays.toString(get()));
		return b.toString();
	}
	
	public void finalize(){
		free(address);
	}
	
	private native long init(float[] data, int[] dims);
	
	private native void free(long address);
	
	private native int[] dims(long src);
	
	private native void reshape(long src, int... d);

	private native float get(long src, int... d);

	private native float[] get(long src);
	
	private native void set(long src, float v, int... d);

	private native void set(long src, float[] data);
	
	private native void fill(long src, float v);

	private native void rand(long src);

	private native void randn(long src);
	
	private native void bernoulli(long src, float p);

	private native long copyInto(long src, long other);

	private native long narrow(long src, int dim, int index, int size);

	private native long select(long src, int dim, int index);

	private native long transpose(long src, int d1, int d2);

	private native long diag(long src, long dst);
	
	private native boolean equals(long src, long other, float threshold);
}
