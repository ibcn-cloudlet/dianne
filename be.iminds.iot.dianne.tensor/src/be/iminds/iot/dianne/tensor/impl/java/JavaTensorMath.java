package be.iminds.iot.dianne.tensor.impl.java;

import java.util.stream.IntStream;

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
	
	private JavaTensor addconvolution2D(JavaTensor res, JavaTensor mat, JavaTensor mat1, JavaTensor mat2, int stride_x, int stride_y, int mode, boolean flip) {
		JavaTensor kernel = mat2;
		if(flip){
			kernel = factory.createTensor(mat2.dims);
			int k = kernel.data.length-1;
			if(mat2.indices==null)
				for(int i=0;i<kernel.data.length;i++)
					kernel.data[i] = mat2.data[k--];
			else
				for(int i=0;i<kernel.data.length;i++)
					kernel.data[i] = mat2.data[ mat2.indices[k--]];
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
				
				for(int i=0;i<r.dims[0];i++){
					int index = (k+i*stride_y)*skip+l;
					for(int j=0;j<r.dims[1];j++){
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
	public JavaTensor spatialconvolve(JavaTensor res, JavaTensor add,
			JavaTensor mat, JavaTensor k, int sx, int sy, int px, int py) {
		int noOutputPlanes = k.size(0);
		int noInputPlanes = k.size(1);
		int kernelHeight = k.size(2);
		int kernelWidth = k.size(3);
		
		int[] outDims = new int[3];
		outDims[0] = noOutputPlanes;
		if(mat.dim()==2){
			outDims[1] = (mat.size(0) + 2*py - kernelHeight)/sy + 1;
			outDims[2] = (mat.size(1) + 2*px - kernelWidth)/sx + 1;
		} else if(mat.dim()==3){
			outDims[1] = (mat.size(1) + 2*py - kernelHeight)/sy + 1;
			outDims[2] = (mat.size(2) + 2*py - kernelWidth)/sx + 1;
		} // else error?
		if(res==null || !res.hasDim(outDims)){
			res = factory.createTensor(outDims);
		}
	
		final JavaTensor output = res;
	
		final JavaTensor input = px !=0 || py !=0 ? factory.getTensorMath().zeropad(null, mat, 0, py, px) : mat;
		
		IntStream
		.range(0, noOutputPlanes)
		.parallel()
		.forEach(i -> {
			JavaTensor planeKernels = k.select(0, i);
			JavaTensor outputPlane = output.select(0, i);
			outputPlane.fill(0.0f);
			
			if(noInputPlanes == 1) {
				JavaTensor kernel = planeKernels.select(0, 0);

				addconvolution2D(outputPlane, outputPlane, input, 
								kernel, sx, sy, 0, false);
			} else {
				for (int j = 0; j < noInputPlanes; j++) {
					JavaTensor kernel = planeKernels.select(0, j);
	
					addconvolution2D(outputPlane, outputPlane, input.select(0, j), 
									kernel, sx, sy, 0, false);
				}
			}

			// add bias
			add(outputPlane, outputPlane, add.get(i));
		});
		
		return res;
	}
	
	@Override
	public JavaTensor spatialdinconvolve(JavaTensor res,
			JavaTensor g, JavaTensor k, int sx, int sy, int px, int py) {
		if(sx!=1 || sy!=1 || px != 0 || py != 0){
			// TODO also implement this for strides != 1 & pads != 0
			throw new UnsupportedOperationException();
		}
		
		int[] outputDims = g.dims();
		int[] kernelDims = k.dims();
		int[] inputDims = {kernelDims[1], outputDims[1] + kernelDims[2] - 1, outputDims[2] + kernelDims[3] - 1};
		
		// backward based on http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
		if(res == null || !res.hasDim(inputDims)){
			res = factory.createTensor(inputDims);
		}
		
		// TODO create subtensors once and reuse?
		for(int i=0;i<inputDims[0];i++){
			JavaTensor planeKernels = k.select(1, i);
			JavaTensor gradInputPlane = res.select(0, i);
			gradInputPlane.fill(0.0f);
			for(int j=0;j<outputDims[0];j++){
				JavaTensor kernel = planeKernels.select(0, j);
				
				// update gradInput
				// this should be "full" convolution and flipped kernel?
				addconvolution2D(gradInputPlane, gradInputPlane,
						g.select(0, j), kernel, 1, 1, 1, true);
			}
		}
		
		return res;
	}
	
	@Override
	public JavaTensor spatialdkerconvolve(JavaTensor res, JavaTensor add,
			JavaTensor g, JavaTensor t, int sx, int sy, int px, int py) {
		if(sx!=1 || sy!=1 || px != 0 || py != 0){
			// TODO also implement this for strides != 1 & pads != 0
			throw new UnsupportedOperationException();
		}
		
		int[] outputDims = g.dims();
		int[] inputDims = t.dims();
		int[] kernelDims = {outputDims[0], inputDims.length == 2 ? 1 : inputDims[0], inputDims[1] - outputDims[1] + 1, inputDims[2] - outputDims[2] + 1};
		
		if(add != null && add != res)
			add.copyInto(res);
		else if(res == null || !res.hasDim(kernelDims)){
			res = factory.createTensor(kernelDims);
		}
		
		// calculate grad weights based on http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
		int noOutputPlanes = outputDims[0];
		int noInputPlanes = inputDims.length == 2 ? 1 : inputDims[0];
		
		for(int i=0;i<noOutputPlanes;i++){
			JavaTensor planeGradKernels = res.select(0, i);
		
			for(int j=0;j<noInputPlanes;j++){
				JavaTensor gradKernel = planeGradKernels.select(0, j);
				
				//  update gradKernel
				addconvolution2D(gradKernel, gradKernel, 
						noInputPlanes== 1 ? t : t.select(0, j), g.select(0, i), 1, 1, 0, false);
			}
		}
		
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
	
	private JavaTensor maxpool2D(JavaTensor res, JavaTensor mat1, int w, int h, int stride_x, int stride_y) {
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
	
	private JavaTensor dmaxpool2D(JavaTensor res, JavaTensor mat2, JavaTensor mat1, int w, int h, int stride_x, int stride_y) {
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
			maxpool2D(pooled.select(0, i), t.select(0,i), w, h, stride_x, stride_y);
		});
		
		return res;
	}
	
	@Override
	public JavaTensor spatialdmaxpool(JavaTensor res, JavaTensor t2, JavaTensor t1, int w, int h, int stride_x, int stride_y) {
		if(res==null || !res.hasDim(t1.dims)){
			res = factory.createTensor(t1.dims);
		}
		
		int noPlanes = t1.dims[0];
		
		final JavaTensor dpooled = res;
		IntStream
		.range(0, noPlanes)
		.parallel()
		.forEach(i -> {
			dmaxpool2D(dpooled.select(0, i), t2.select(0, i), t1.select(0, i), w, h, stride_x, stride_y);
		});
		
		return res;
	}

	@Override
	public JavaTensor scale2D(JavaTensor res, JavaTensor t, int... dims) {
		if(res == null){
			res = factory.createTensor(dims);
		}
		
		// TODO what for n-d tensors
		int y_in = t.dims[t.dims.length-2];
		int x_in = t.dims[t.dims.length-1];
		
		int y_out = dims[dims.length-2];
		int x_out = dims[dims.length-1];
		
		float s_y = (y_in-1)/(float)(y_out-1);
		float s_x = (x_in-1)/(float)(x_out-1); 

		int c_in = t.dims.length == 3 ? t.dims[0] : 1;
		int c_out = dims.length == 3 ? dims[0] : 1;
		
		float yy,xx;
		int x1,x2,y1,y2, cc;
		int i1, i2, i3, i4;
		float v1,v2,v3,v4;
		float dx,dy;
		float r;
		
		for(int c=0;c<c_out;c++){
			for(int y=0;y<y_out;y++){
				for(int x=0;x<x_out;x++){
					cc = c;
					if(cc >= c_in){
						cc = 0;
					}
					
					yy = y*s_y;
					xx = x*s_x;
					
					// bilinear interpolation
					x1 = (int)xx;
					x2 = x1+1;
					if(x2==x_in)
						x2--;
					y1 = (int)yy;
					y2 = y1+1;
					if(y2==y_in)
						y2--;
					
					i1 = x_in*y_in*cc + x_in*y1+x1;
					i2 = x_in*y_in*cc + x_in*y1+x2;
					i3 = x_in*y_in*cc + x_in*y2+x1;
					i4 = x_in*y_in*cc + x_in*y2+x2;
					
					v1 = t.data[(t.indices==null? i1 : t.indices[i1])];
					v2 = t.data[(t.indices==null? i2 : t.indices[i2])];
					v3 = t.data[(t.indices==null? i3 : t.indices[i3])];
					v4 = t.data[(t.indices==null? i4 : t.indices[i4])];
					
					dx = xx-x1;
					dy = yy-y1;
					
					r = v1*(1-dy)*(1-dx)
							 + v2 * (1-dy)*(dx)
							 + v3 * (dy)*(1-dx)
							 + v4 * (dx)*(dy);
					
					// nearest neighbor
	//				int xr = Math.round(xx);
	//				int yr = Math.round(yy);
	//				if(xr==x_in)
	//					xr--;
	//				if(yr==y_in)
	//					yr--;
	//				float r = t.data[x_in*yr+xr];
					
					res.data[x_out*y_out*c + x_out*y + x] = r;
				}
			}
		}
		
		return res;
	}
}
