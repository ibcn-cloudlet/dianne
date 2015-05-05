package be.iminds.iot.dianne.tensor.impl.java;

import java.util.Arrays;
import java.util.stream.IntStream;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath<JavaTensor> {

	private final TensorFactory<JavaTensor> factory;
	
	public JavaTensorMath(TensorFactory<JavaTensor> factory) {
		this.factory = factory;
	}

	@FunctionalInterface
	private interface Operator {
		public float apply(final float... params);
	}
	
	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t.dims);
		}
		
		for(int i=0;i<(res.indices==null ? res.data.length : res.indices.length);i++){
			res.data[(res.indices==null ? i :res.indices[i])] 
					= op.apply(t.data[(t.indices==null ? i: t.indices[i])]);
		}
		
		return res;
	}

	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t1, final JavaTensor t2){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t1.dims);
		}

		for(int i=0;i<(t1.indices==null? t1.data.length : t1.indices.length);i++){
			res.data[(res.indices==null? i : res.indices[i])] 
					= op.apply(t1.data[(t1.indices==null? i : t1.indices[i])], 
							t2.data[(t2.indices==null? i : t2.indices[i])]);
		}
		
		return res;
	}
	
	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t1, final JavaTensor t2, final JavaTensor t3){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t1.dims);
		}

		for(int i=0;i<(t1.indices==null? t1.data.length : t1.indices.length);i++){
			res.data[(res.indices==null? i : res.indices[i])] 
					= op.apply(t1.data[(t1.indices==null? i : t1.indices[i])], 
							t2.data[(t2.indices==null? i : t2.indices[i])],
							t3.data[(t3.indices==null? i : t3.indices[i])]);
		}
		
		return res;
	}
	
	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t1, final JavaTensor t2, final JavaTensor t3, final JavaTensor t4){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t1.dims);
		}

		for(int i=0;i<(t1.indices==null? t1.data.length : t1.indices.length);i++){
			res.data[(res.indices==null? i : res.indices[i])] 
					= op.apply(t1.data[(t1.indices==null? i : t1.indices[i])], 
							t2.data[(t2.indices==null? i : t2.indices[i])],
							t3.data[(t3.indices==null? i : t3.indices[i])],
							t4.data[(t4.indices==null? i : t4.indices[i])]);
		}
		
		return res;
	}
	
	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor, final float value) {
		return apply(res, params ->  params[0] + value, tensor);
	}

	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		return apply(res, params -> params[0] + params[1], tensor1, tensor2);
	}

	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor1, final float value, final JavaTensor tensor2) {
		return apply(res, params -> params[0] + value * params[1], tensor1, tensor2);
	}

	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor, final float value) {
		return apply(res, params -> params[0] - value, tensor);
	}

	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		return apply(res, params -> params[0] - params[1], tensor1, tensor2);
	}
	
	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor1, final float value, final JavaTensor tensor2) {
		return apply(res, params -> params[0] - value * params[1], tensor1, tensor2);
	}
	
	@Override
	public JavaTensor mul(JavaTensor res, final JavaTensor tensor, final float value) {
		return apply(res, params -> params[0] * value, tensor);
	}

	@Override
	public JavaTensor cmul(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		return apply(res, params -> params[0] * params[1], tensor1, tensor2);
	}

	@Override
	public JavaTensor div(JavaTensor res, final JavaTensor tensor, final float value) {
		return apply(res, params -> params[0] / value, tensor);
	}
	
	@Override
	public JavaTensor cdiv(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		return apply(res, params -> params[0] / params[1], tensor1, tensor2);
	}

	@Override
	public float dot(final JavaTensor vec1, final JavaTensor vec2) {
		// TODO check dimensions
		float dot = 0;
		
		for(int i=0;i<(vec1.indices==null? vec1.data.length : vec1.indices.length);i++){
			dot+= vec1.data[(vec1.indices==null? i : vec1.indices[i])] 
					* vec2.data[(vec2.indices==null? i: vec2.indices[i])];
		}
		
		return dot;
	}
	
	@Override
	public JavaTensor vv(JavaTensor res, final JavaTensor vec1, final JavaTensor vec2) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(vec1.size(), vec2.size());
		}
		for(int i=0;i<vec1.size();i++){
			for(int j=0;j<vec2.size();j++){
				res.set(vec1.get(i)*vec2.get(j), i, j);
			}
		}
		return res;
	}

	@Override
	public JavaTensor mv(JavaTensor res, final JavaTensor mat, final JavaTensor vec) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(mat.dims[0]);
		}
		
		int k = 0;
		for(int i=0;i<(mat.indices==null? mat.data.length : mat.indices.length);){
			float v = 0;
			for(int j=0;j<(vec.indices==null ? vec.data.length : vec.indices.length);j++){
				v+= mat.data[(mat.indices==null? i++ : mat.indices[i++])]
						*vec.data[(vec.indices==null? j : vec.indices[j])];
			}
			res.data[(res.indices==null? k : res.indices[k])] = v;
			k++;
		}
	
		return res;
	}
	
	@Override
	public JavaTensor tmv(JavaTensor res, final JavaTensor mat, final JavaTensor vec) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(mat.dims[1]);
		}
		res.fill(0.0f);
		int j = 0;
		int k = 0;
		for(int i=0;i<(mat.indices==null? mat.data.length : mat.indices.length);i++){
			if(j==(res.indices==null? res.data.length : res.indices.length)){
				j = 0;
				k++;
			}
			res.data[(res.indices==null? j++ : res.indices[j++])] 
					+= mat.data[(mat.indices==null? i : mat.indices[i])]
							*vec.data[(vec.indices==null? k : vec.indices[k])];
		}
		return res;
	}

	@Override
	public JavaTensor mm(JavaTensor res, final JavaTensor mat1, final JavaTensor mat2) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(mat1.dims[0],mat2.dims[1]);
		}
		for(int k=0;k<mat2.dims[1];k++){
			for(int i=0;i<mat1.dims[0];i++){
				float v = 0;
				for(int j=0;j<mat1.dims[1];j++){
					v+= mat1.get(i,j) * mat2.get(j,k);
				}
				res.set(v, i, k);
			}
		}
		return res;
	}
	
	@Override
	public JavaTensor addvv(JavaTensor res, final JavaTensor mat, final JavaTensor vec1,
			final JavaTensor vec2) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(vec1.size(), vec2.size());
		}
		int k = 0;
		int j = 0;
		for(int i=0;i<(mat.indices==null? mat.data.length : mat.indices.length);i++){
			if(j==(vec2.indices==null? vec2.data.length : vec2.indices.length)){
				j = 0;
				k++;
			}
			res.data[(res.indices==null? i : res.indices[i])] 
					= mat.data[(mat.indices==null? i : mat.indices[i])] 
							+ vec1.data[(vec1.indices==null ? k : vec1.indices[k])]
							*vec2.data[(vec2.indices==null ? j++ : vec2.indices[j++])];
		}
		return res;
	}


	@Override
	public JavaTensor addmv(JavaTensor res, final JavaTensor vec1, final JavaTensor mat,
			final JavaTensor vec2) {
		if(res==null){
			res = factory.createTensor(mat.dims[0]);
		}
		
		int k = 0;
		for(int i=0;i<(mat.indices==null ? mat.data.length : mat.indices.length);){
			float v = 0;
			for(int j=0;j<(vec2.indices==null? vec2.data.length : vec2.indices.length);j++){
				v+= mat.data[(mat.indices==null? i++ : mat.indices[i++])]
						*vec2.data[(vec2.indices==null? j : vec2.indices[j])];
			}
			res.data[(res.indices==null?  k : res.indices[k])] 
					= v + vec1.data[(vec1.indices==null ? k : vec1.indices[k])];
			k++;
		}
		return res;
	}

	@Override
	public JavaTensor addmm(JavaTensor res, final JavaTensor mat, final JavaTensor mat1,
			final JavaTensor mat2) {
		res = mm(res, mat1, mat2);
		return add(res, mat, res);
	}

	@Override
	public JavaTensor exp(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> (float) Math.exp(params[0]), tensor);
	}

	@Override
	public JavaTensor log(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> (float) Math.log(params[0]), tensor);
	}
	
	@Override
	public JavaTensor tanh(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> (float) Math.tanh(params[0]), tensor);
	}

	@Override
	public JavaTensor sigmoid(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> (float)(1.0/(1+Math.exp(-params[0]))), tensor);
	}

	@Override
	public JavaTensor thresh(JavaTensor res, final JavaTensor tensor, final float thresh, final float coeff, final float offset) {
		return apply(res, params -> params[0] > thresh ? params[0] : coeff * params[0] + offset, tensor);
	}
	
	@Override
	public JavaTensor thresh(JavaTensor res, final JavaTensor tensor, final JavaTensor threshs, final JavaTensor coeffs, final JavaTensor offsets) {
		return apply(res, params -> params[0] > params[1] ? params[0] : params[2] * params[0] + params[3], tensor, threshs, coeffs, offsets);
	}
	
	@Override
	public JavaTensor dtanh(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> (1.0f - params[0]*params[0]), tensor);
	}

	@Override
	public JavaTensor dsigmoid(JavaTensor res, final JavaTensor tensor) {
		return apply(res, params -> params[0]*(1.0f - params[0]), tensor);
	}

	@Override
	public JavaTensor dthresh(JavaTensor res, final JavaTensor tensor, final float thresh, final float coeff) {
		return apply(res, params -> params[0] > thresh ? 1 : coeff, tensor);
	}
	
	@Override
	public JavaTensor dthresh(JavaTensor res, final JavaTensor tensor, final JavaTensor threshs, final JavaTensor coeffs) {
		return apply(res, params -> params[0] > params[1] ? 1 : params[2], tensor, threshs, coeffs);
	}

	@Override
	public JavaTensor softmax(JavaTensor res, final JavaTensor tensor){
		final float max = max(tensor);
		res = apply(res, params -> (float) Math.exp(params[0]-max), tensor);
		
		float sum = sum(res);
		res = div(res, res, sum);
		
		return res;
	}
	
	@Override
	public float sum(final JavaTensor tensor) {
		float sum = 0;
		for(int i=0;i<(tensor.indices==null?tensor.data.length : tensor.indices.length);i++){
			sum+= tensor.data[(tensor.indices==null? i : tensor.indices[i])];
		}
		return sum;
	}

	@Override
	public float max(final JavaTensor tensor) {
		float max = -Float.MAX_VALUE;
		for(int i=0;i<(tensor.indices==null?tensor.data.length : tensor.indices.length);i++){
			float val = tensor.data[(tensor.indices==null? i : tensor.indices[i])];
			if(val > max)
				max = val;
		}
		return max;
	}

	@Override
	public float min(final JavaTensor tensor) {
		float min = Float.MAX_VALUE;
		for(int i=0;i<(tensor.indices==null?tensor.data.length : tensor.indices.length);i++){
			float val = tensor.data[(tensor.indices==null? i : tensor.indices[i])];
			if(val < min)
				min = val;
		}
		return min;
	}

	@Override
	public float mean(final JavaTensor tensor) {
		float mean = sum(tensor);
		mean /= tensor.size();
		return mean;
	}

	@Override
	public int argmax(final JavaTensor tensor) {
		float max = -Float.MAX_VALUE;
		int index = -1;
		for(int i=0;i<(tensor.indices==null?tensor.data.length : tensor.indices.length);i++){
			float val = tensor.data[(tensor.indices==null? i : tensor.indices[i])];
			if(val > max){
				max = val;
				index = i;
			}
		}
		return index;
	}
	
	
	@Override
	public int argmin(final JavaTensor tensor) {
		float min = Float.MAX_VALUE;
		int index = -1;
		for(int i=0;i<(tensor.indices==null?tensor.data.length : tensor.indices.length);i++){
			float val = tensor.data[(tensor.indices==null? i : tensor.indices[i])];
			if(val < min){
				min = val;
				index = i;
			}
		}
		return index;
	}

	@Override
	public JavaTensor convolution2D(JavaTensor res, JavaTensor mat1, JavaTensor mat2, int stride_x, int stride_y, int mode, boolean flip) {
		return addconvolution2D(res, null, mat1, mat2, stride_x, stride_y, mode, flip);
	}
	
	@Override
	public JavaTensor addconvolution2D(JavaTensor res, JavaTensor mat, JavaTensor mat1, JavaTensor mat2, int stride_x, int stride_y, int mode, boolean flip) {
		JavaTensor kernel = mat2;
		if(flip){
			kernel = factory.createTensor(mat2.dims);
			int k = kernel.data.length-1;
			for(int i=0;i<kernel.data.length;i++){
				kernel.data[i] = mat2.data[(mat2.indices==null? k-- : mat2.indices[k--])];
			}
		}
		if(mode == 1){
			// full
			JavaTensor padded = factory.createTensor(mat1.dims[0]+(mat2.dims[0]-1)*2, mat1.dims[1]+(mat2.dims[1]-1)*2);
			int offset = (mat2.dims[1]-1) * padded.dims[0] + (mat2.dims[0]-1);
			int skip = (mat2.dims[0]-1)*2;
			int k = offset;
			int l = 0;
			for(int i=0;i<mat1.dims[0];i++){
				for(int j=0;j<mat1.dims[1];j++){
					padded.data[k++] = mat1.data[(mat1.indices==null? l++ : mat1.indices[l++])];
				}
				k+=skip;
			}
			return convValid(res, padded, kernel, mat, stride_x, stride_y);
		} else if(mode == 2){
			// same
			JavaTensor padded = factory.createTensor(mat1.dims[0]+mat2.dims[0]-1, mat1.dims[1]+mat2.dims[1]-1);
			int offset = (mat2.dims[1]-1)/2 * padded.dims[0] + ((mat2.dims[0]-1)/2);
			int skip = mat2.dims[0]-1;
			int k = offset;
			int l = 0;
			for(int i=0;i<mat1.dims[0];i++){
				for(int j=0;j<mat1.dims[1];j++){
					padded.data[k++] = mat1.data[(mat1.indices==null? l++ : mat1.indices[l++])];
				}
				k+=skip;
			}
			return convValid(res, padded, kernel, mat, stride_x, stride_y);
		} else {
			return convValid(res, mat1, kernel, mat, stride_x, stride_y);
		}
	}

	private JavaTensor convValid(JavaTensor res, JavaTensor mat1, JavaTensor mat2, JavaTensor add, int stride_x, int stride_y){
		int h = mat2.dims[0];
		int w = mat2.dims[1];
		
		int y = mat1.dims[0] - h ;
		int x = mat1.dims[1] - w ;
		
		JavaTensor r = res;
		if(r==null){
			r = factory.createTensor(y/stride_y+1, x/stride_x+1);
		} else if(add==null){
			r.fill(0.0f);
		}

		if(add!=null){
			add.copyInto(r);
		}
		
		int a,f = 0;		
		int skip = mat1.dims[1];
		for(int k=0;k<h;k++){
			for(int l=0;l<w;l++){
				a = 0;

				float c = mat2.data[mat2.indices==null? f : mat2.indices[f]];
				
				for(int i=0;i<r.dims[1];i++){
					int index = (k+i*stride_y)*skip+l;
					for(int j=0;j<r.dims[0];j++){
						r.data[(r.indices==null? a : r.indices[a])]
								+= mat1.data[mat1.indices==null ? index : mat1.indices[index]]
										* c;
						
						index+=stride_x;
						a++;
					}
				}
				f++;
			}
		}
		return r;
	}
	
	@Override
	public JavaTensor maxpool2D(JavaTensor res, JavaTensor mat1, int w, int h, int stride_x, int stride_y) {
		int r_h = (mat1.size(0) - h )/stride_y + 1;
		int r_w = (mat1.size(1) - w )/stride_x + 1;
		int skip = mat1.size(1);
		if(res==null){
			res = factory.createTensor(r_h, r_w);
		}
	
		for(int i=0;i<r_h;i++){
			for(int j=0;j<r_w;j++){
				float max = -Float.MAX_VALUE;
				int rindex = i*r_w+j;
				for(int k=0;k<h;k++){
					for(int l=0;l<w;l++){
						int index = (i*stride_y+k)*skip+(j*stride_x+l);
						float val = mat1.data[(mat1.indices==null? index : mat1.indices[index])];
						if(val>max)
							max = val;
					}
				}
				res.data[(res.indices==null? rindex : res.indices[rindex])] = max;
			}
		}

		return res;
	}
	
	@Override
	public JavaTensor dmaxpool2D(JavaTensor res, JavaTensor mat2, JavaTensor mat1, int w, int h, int stride_x, int stride_y) {
		int r_h = (mat1.size(0) - h )/stride_y + 1;
		int r_w = (mat1.size(1) - w )/stride_x + 1;
		int skip = mat1.size(1);
		
		if(res==null){
			res = factory.createTensor(mat1.size(0), mat1.size(1));
		}
	
		for(int i=0;i<r_h;i++){
			for(int j=0;j<r_w;j++){
				float max = -Float.MAX_VALUE;
				int maxIndex = -1;
				int rindex = i*r_w+j;
				for(int k=0;k<h;k++){
					for(int l=0;l<w;l++){
						int index = (i*stride_y+k)*skip+(j*stride_x+l);
						float val = mat1.data[(mat1.indices==null? index : mat1.indices[index])];
						if(val>max){
							max = val;
							maxIndex  = index;
						}
					}
				}
				res.data[(res.indices==null? maxIndex : res.indices[maxIndex])] = 
						mat2.data[(mat2.indices==null? rindex : mat2.indices[rindex])];
			}
		}

		return res;
	}

	@Override
	public JavaTensor spatialconvolve(JavaTensor res, JavaTensor add,
			JavaTensor mat, JavaTensor k, int sx, int sy) {
		int noOutputPlanes = k.size(0);
		int noInputPlanes = k.size(1);
		int kernelHeight = k.size(2);
		int kernelWidth = k.size(3);
		
		int[] outDims = new int[3];
		outDims[0] = noOutputPlanes;
		if(mat.dim()==2){
			outDims[1] = (mat.size(0) - kernelHeight)/sy + 1;
			outDims[2] = (mat.size(1) - kernelWidth)/sx + 1;
		} else if(mat.dim()==3){
			outDims[1] = (mat.size(1) - kernelHeight)/sy + 1;
			outDims[2] = (mat.size(2) - kernelWidth)/sx + 1;
		} // else error?
		if(res==null || !res.hasDim(outDims)){
			res = factory.createTensor(outDims);
		}
		
		final JavaTensor output = res;
		
		IntStream
		.range(0, noOutputPlanes)
		.parallel()
		.forEach(i -> {
			JavaTensor planeKernels = k.select(0, i);
			JavaTensor outputPlane = output.select(0, i);
			outputPlane.fill(0.0f);

			for (int j = 0; j < noInputPlanes; j++) {
				JavaTensor kernel = planeKernels.select(0, j);

				addconvolution2D(outputPlane, outputPlane, noInputPlanes == 1 ? mat : mat.select(0, j), 
								kernel, sx, sy, 0, false);
			}

			// add bias
			add(outputPlane, outputPlane, add.get(i));
		});
		
		return res;
	}

	@Override
	public JavaTensor zeropad(JavaTensor res, JavaTensor t, int... paddings) {
		
		int[] paddedDims = new int[t.dims.length];
		for(int i=0;i<paddedDims.length;i++){
			paddedDims[i] = t.dims[i] + paddings[i]*2;
		}
		if(res==null || !res.hasDim(paddedDims)){
			res = factory.createTensor(paddedDims);
		}
		
		int[] ranges = new int[paddedDims.length*2];
		for(int i=0;i<paddedDims.length;i++){
			ranges[i*2] = paddings[i];
			ranges[i*2+1] = t.dims[i];
		}
		JavaTensor sub = res.narrow(ranges);
		t.copyInto(sub);
		
		return res;
	}

	@Override
	public JavaTensor spatialmaxpool(JavaTensor res, JavaTensor t, int w, int h, int stride_x, int stride_y) {
		int noPlanes = t.size(0);
		// requires 3d input tensor?
		int y = (t.size(1) - h )/stride_y + 1;
		int x = (t.size(2) - w )/stride_x + 1;
		if(res==null || !res.hasDim(noPlanes, y, x)){
			res = factory.createTensor(noPlanes, y, x);
		}
		
		final JavaTensor pooled = res;
		IntStream
		.range(0, noPlanes)
		.parallel()
		.forEach(i -> {
			factory.getTensorMath().maxpool2D(pooled.select(0, i), t.select(0,i), w, h, stride_x, stride_y);
		});
		
		return res;
	}
}
