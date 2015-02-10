package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath<JavaTensor> {

	private final TensorFactory<JavaTensor> factory;
	
	public JavaTensorMath(TensorFactory<JavaTensor> factory) {
		this.factory = factory;
	}

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
	
	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor, final float value) {
		Operator add = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] + value;
			}
		};
		return apply(res, add, tensor);
	}

	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		Operator add = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] + params[1];
			}
		};
		return apply(res, add, tensor1, tensor2);
	}

	@Override
	public JavaTensor add(JavaTensor res, final JavaTensor tensor1, final float value, final JavaTensor tensor2) {
		Operator add = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] + value * params[1];
			}
		};
		return apply(res, add, tensor1, tensor2);
	}

	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor, final float value) {
		Operator sub = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] - value;
			}
		};
		return apply(res, sub, tensor);
	}

	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		Operator sub = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] - params[1];
			}
		};
		return apply(res, sub, tensor1, tensor2);
	}
	
	@Override
	public JavaTensor sub(JavaTensor res, final JavaTensor tensor1, final float value, final JavaTensor tensor2) {
		Operator sub = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] - value * params[1];
			}
		};
		return apply(res, sub, tensor1, tensor2);
	}
	
	@Override
	public JavaTensor mul(JavaTensor res, final JavaTensor tensor, final float value) {
		Operator mul = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] * value;
			}
		};
		return apply(res, mul, tensor);
	}

	@Override
	public JavaTensor cmul(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		Operator cmul = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] * params[1];
			}
		};
		return apply(res, cmul, tensor1, tensor2);
	}

	@Override
	public JavaTensor div(JavaTensor res, final JavaTensor tensor, final float value) {
		Operator div = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] / value;
			}
		};
		return apply(res, div, tensor);
	}

	@Override
	public JavaTensor div(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		Operator div = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] / params[1];
			}
		};
		return apply(res, div, tensor1, tensor2);
	}
	
	@Override
	public JavaTensor cdiv(JavaTensor res, final JavaTensor tensor1, final JavaTensor tensor2) {
		Operator cdiv = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] / params[1];
			}
		};
		return apply(res, cdiv, tensor1, tensor2);
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
		Operator exp = new Operator(){
			@Override
			public float apply(float... params) {
				return (float) Math.exp(params[0]);
			}
		};
		return apply(res, exp, tensor);
	}

	@Override
	public JavaTensor log(JavaTensor res, final JavaTensor tensor) {
		Operator log = new Operator(){
			@Override
			public float apply(float... params) {
				return (float) Math.log(params[0]);
			}
		};
		return apply(res, log, tensor);
	}
	
	@Override
	public JavaTensor tanh(JavaTensor res, final JavaTensor tensor) {
		Operator tanh = new Operator(){
			@Override
			public float apply(float... params) {
				return (float) Math.tanh(params[0]);
			}
		};
		return apply(res, tanh, tensor);
	}

	@Override
	public JavaTensor sigmoid(JavaTensor res, final JavaTensor tensor) {
		Operator sigmoid = new Operator(){
			@Override
			public float apply(float... params) {
				return (float)(1.0/(1+Math.exp(-params[0])));
			}
		};
		return apply(res, sigmoid, tensor);
	}

	@Override
	public JavaTensor thresh(JavaTensor res, final JavaTensor tensor, final float thresh, final float val) {
		Operator threshOp = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] > thresh ? params[0] : val;
			}
		};
		return apply(res, threshOp, tensor);
	}
	
	@Override
	public JavaTensor dtanh(JavaTensor res, final JavaTensor tensor) {
		Operator dtanh = new Operator(){
			@Override
			public float apply(float... params) {
				return (1.0f - params[0]*params[0]);
			}
		};
		return apply(res, dtanh, tensor);
	}

	@Override
	public JavaTensor dsigmoid(JavaTensor res, final JavaTensor tensor) {
		Operator dsigmoid = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0]*(1.0f - params[0]);
			}
		};
		return apply(res, dsigmoid, tensor);
	}

	@Override
	public JavaTensor dthresh(JavaTensor res, final JavaTensor tensor, final float thresh) {
		Operator threshOp = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] > thresh ? 1 : 0;
			}
		};
		return apply(res, threshOp, tensor);
	}
	
	@Override
	public JavaTensor softmax(JavaTensor res, final JavaTensor tensor){
		final float max = max(tensor);
		Operator op = new Operator(){
			@Override
			public float apply(float... params) {
				return (float) Math.exp(params[0]-max);
			}
		};
		res = apply(res, op, tensor);
		
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
	public JavaTensor convolution2D(JavaTensor res, JavaTensor mat1, JavaTensor mat2) {
		// TODO stride?
		int h = (mat2.size(0)-1)/2;
		int w = (mat2.size(1)-1)/2;

		int y = mat1.size(0) - 2*h;
		int x = mat1.size(1) - 2*w;
		
		int skip = mat1.size(0);
		
		if(res==null){
			res = factory.createTensor(y, x);
		}
		
		// TODO check dims?
		int a = 0;
		for(int i=0;i<y;i++){
			for(int j=0;j<x;j++){
				float r = 0;
				int f = 0;
				for(int k=-h;k<=h;k++){
					for(int l=-w;l<=w;l++){
						int index = (i+k+h)*skip+(j+l+w);
						r += mat1.data[mat1.indices==null ? index : mat1.indices[index]]
								* mat2.data[mat2.indices==null? f++ : mat2.indices[f++]];
					}
				}
				res.data[(res.indices==null? a++ : res.indices[a++])] = r;
			}
		}
		
		return res;
	}

}
