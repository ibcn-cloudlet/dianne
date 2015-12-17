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
package be.iminds.iot.dianne.tensor.impl.java;

import java.util.Arrays;
import java.util.Random;

import be.iminds.iot.dianne.tensor.Tensor;

public class JavaTensor implements Tensor<JavaTensor> {

	static Random random = new Random(1234);
	
	int[] dims;
	float[] data;
	int[] strides;
	int[] indices = null;
	
	JavaTensor(final int... d){
		init(null, d);
	
		data = new float[size()];
	}

	JavaTensor(final float[] data, final int ... dims){
		init(data, dims);
	}
	
	private void init(final float[] data, final int[] dims){
		this.data = data;

		reshape(dims);
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
	public void reshape(final int... d){
		// TODO check whether to total data length is ok with these dimensions?
		this.dims = d;
		this.strides = new int[dims.length];
		int stride = 1;
		for(int i=dims.length-1;i>=0;i--){
			strides[i] = stride;
			stride*= dims[i];
		}
	}
	
	@Override
	public float get(final int... d) {
		int[] i = d;
		if(d.length!=dims.length){
			i = new int[dims.length];
			int l = 0;
			for(int k=0;k<dims.length;k++){
				if(dims[k]==1 || l >= d.length){
					i[k] = 0;
				} else {
					i[k] = d[l];
					l++;
				}
			}
		}
		
		int index = getIndex(i);
		return data[index];
	}
	
	@Override
	public float[] get(){
		float[] result = new float[size()];
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			result[i] = data[(indices==null ? i : indices[i])];
		}
		return result;
	}
	
	@Override
	public void set(final float v, final int... d) {
		assert d.length == dims.length;
		
		int index = getIndex(d);
		data[index] = v;
	}
	
	@Override
	public void set(final float[]  d) {
		assert d.length == data.length;
		
		System.arraycopy(d, 0, data, 0, data.length);
	}
	
	@Override
	public String toString(){
		if(dims.length == 1){
			// format as vector
			String s = "[";
			for(int i=0;i< (indices==null? data.length : indices.length);i++){
				s+=data[(indices==null ? i : indices[i])]+", ";
			}
			s = s.substring(0, s.length()-2);
			s+="]";
			return s;
		} else if(dims.length == 2){
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
		} else {
			String s = "[";
			for(int i=0;i< (indices==null? data.length : indices.length);i++){
				if(i>0 && i%dims[dims.length-1]==0)
					s+="\n";
				if(i>0 && i%(dims[dims.length-1]*dims[dims.length-2])==0)
					s+="\n";
				
				s+=data[(indices==null ? i : indices[i])]+", ";
			}
			s = s.substring(0, s.length()-2);
			s+="]";
			return s;
		}
	}

	@Override
	public void fill(final float v) {
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			data[(indices==null ? i : indices[i])] = v;
		}
	}

	@Override
	public void rand() {
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			data[(indices==null ? i : indices[i])] = random.nextFloat();
		}
	}
	
	@Override
	public void randn() {
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			data[(indices==null ? i : indices[i])] = (float) random.nextGaussian();
		}
	}
	
	@Override
	public void bernoulli(float p) {
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			float f = random.nextFloat();
			float b = f > p ? 0 : 1;
			data[(indices==null ? i : indices[i])] = b;
		}
	}

	@Override
	public boolean equals(Object other){
		if(!(other instanceof JavaTensor)){
			return false;
		} 
		JavaTensor o = (JavaTensor) other;
		
		return equals(o);

	}
	
	@Override
	public boolean equals(JavaTensor o){
		if(!equalSize(o)){
			return false;
		}
		
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			if(data[(indices==null ? i : indices[i])] 
					!= o.data[(o.indices==null ? i : o.indices[i])]){
				return false;
			}
		}

		return true;
	}
	
	@Override
	public boolean equals(JavaTensor o, float threshold){
		if(!equalSize(o)){
			return false;
		}
		
		for(int i=0;i< (indices==null? data.length : indices.length);i++){
			float diff = data[(indices==null ? i : indices[i])] - 
					o.data[(o.indices==null ? i : o.indices[i])];
			diff = diff < 0 ? -diff : diff;
			if(diff > threshold){
				return false;
			}
		}

		return true;
	}

	private boolean equalSize(JavaTensor o){
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
		return true;
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
	public JavaTensor copyInto(JavaTensor other) {
		if(other == this)
			return this;
		
		if(other == null
				|| this.size() != other.size())
			other = new JavaTensor(dims);
		
		for(int i=0;i<(indices==null?data.length:indices.length);i++){
			other.data[(other.indices==null ? i : other.indices[i])] 
					= data[(indices==null ? i : indices[i])];
		}
		
		return other;
	}

	@Override
	public JavaTensor narrow(final int dim, final int index, final int size) {
		// create new tensor with same data and dims
		JavaTensor narrow = new JavaTensor(data, dims);
		
		// generate indices for the narrowed dims and offsets
		int[] narrowDims = dims.clone();
		narrowDims[dim] = size;
		int[] offsets = new int[dims.length];
		offsets[dim] = index;
		
		narrow.indices = generateIndices(offsets, narrowDims);
		narrow.reshape(narrowDims);
		
		return narrow;
	}

	@Override
	public JavaTensor narrow(final int... ranges) {
		// create new tensor with same data and dims
		JavaTensor narrow = new JavaTensor(data, dims);
				
		// generate indices for the narrowed dims and offsets
		int[] narrowDims = dims.clone();
		int[] offsets = new int[dims.length];
		for(int i=0;i<ranges.length/2;i++){
			offsets[i] = ranges[2*i];
			narrowDims[i] = ranges[2*i+1];
		}
		
		narrow.indices = generateIndices(offsets, narrowDims);
		narrow.reshape(narrowDims);
		
		return narrow;
	}

	@Override
	public JavaTensor select(final int dim, final int index) {
		JavaTensor result = narrow(dim, index, 1); 
		int[] newDims = new int[result.dims.length-1];
		int k = 0;
		for(int i=0;i<result.dims.length;i++){
			if(i!=dim){
				newDims[k++] = result.dims[i];
			}
		}
		// fix strides
		result.reshape(newDims);
		return result;
	}
	
	@Override
	public JavaTensor transpose(JavaTensor res, final int d1, final int d2) {
		if(this!=res)
			res = this.copyInto(res);
		
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
	
	@Override
	public JavaTensor diag(JavaTensor res) {
		// TODO check if tensor is nxn matrix?
		if(res==null){
			res = new JavaTensor(dims[0]);
		}
		for(int i=0;i<dims[0];i++){
			res.set(get(i,i), i);
		}
		return res;
	}
	
	int getIndex(final int[] d){
		int index = 0;
		for(int i=0;i<d.length;i++){
			index += strides[i]*d[i];
		}
		return indices==null? index : indices[index];
	}
	

	private int[] generateIndices(int[] offsets, int[] dims){
		IndexGenerator it = new IndexGenerator(offsets, dims);
		
		int size = 1;
		for(int k=0;k<dims.length;k++){
			size *= dims[k];
		}
		int[] newIndices = new int[size];
		
		int i = 0;
		while(it.hasNext()){
			int index = it.next();
			newIndices[i++] = indices==null ? index : indices[index]; 
		}
		return newIndices;
	}
	
	class IndexGenerator {
		private int[] dims;
		private int[] index; // 3D index
		private int current = 0;
		private int next; // linear index
		
		public IndexGenerator(int[] offsets, int[] dims){
			// new target dims to generate indices for
			this.dims = dims;
			// start at offset with current strides
			this.index = new int[dims.length];
			next = 0;
			for(int i=0;i<offsets.length;i++){
				next += strides[i]*offsets[i];
			}		
		}
		
		public int next(){
			current = next;
			boolean incremented = false;
			int dim = dims.length-1;
			while(!incremented){
				index[dim]++;
				if(index[dim]==dims[dim]){
					index[dim] = 0;
					next-=strides[dim]*(dims[dim]-1);
					dim--;
				} else {
					incremented = true;
					next+= strides[dim];
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
