package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensor.JavaTensorIterator;

public class JavaTensorMath implements TensorMath<JavaTensor> {

	private final TensorFactory<JavaTensor> factory;
	
	public JavaTensorMath(TensorFactory<JavaTensor> factory) {
		this.factory = factory;
	}

	private interface Operator {
		public float apply(float... params);
	}
	
	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t.dims);
		}
		JavaTensorIterator resIt = res.iterator();
		JavaTensorIterator tIt = t.iterator();
		while(tIt.hasNext()){
			res.data[resIt.next()] = op.apply(t.data[tIt.next()]);
		}
		return res;
	}

	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t1, final JavaTensor t2){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t1.dims);
		}
		
		JavaTensorIterator resIt = res.iterator();
		JavaTensorIterator t1It = t1.iterator();
		JavaTensorIterator t2It = t2.iterator();
		while(t1It.hasNext()){
			res.data[resIt.next()] = op.apply(t1.data[t1It.next()], t2.data[t2It.next()]);
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
	public JavaTensor sub(JavaTensor res, JavaTensor tensor, float value) {
		Operator sub = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] - value;
			}
		};
		return apply(res, sub, tensor);
	}

	@Override
	public JavaTensor sub(JavaTensor res, JavaTensor tensor1, JavaTensor tensor2) {
		Operator sub = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0] - params[1];
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
		
		JavaTensorIterator v1It = vec1.iterator();
		JavaTensorIterator v2It = vec2.iterator();
		while(v1It.hasNext()){
			dot+= vec1.data[v1It.next()] * vec2.data[v2It.next()];
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
		JavaTensorIterator it = mat.iterator();
		JavaTensorIterator rIt = res.iterator();
		while(it.hasNext()){
			float v = 0;
			JavaTensorIterator vIt = vec.iterator();
			while(vIt.hasNext()){
				v+= mat.data[it.next()]*vec.data[vIt.next()];
			}
			res.data[rIt.next()] = v;
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
		JavaTensorIterator mIt = mat.iterator();
		JavaTensorIterator rIt = res.iterator();
		JavaTensorIterator vIt = vec.iterator();
		int vi = vIt.next();
		while(mIt.hasNext()){
			if(!rIt.hasNext()){
				rIt = res.iterator();
				vi = vIt.next();
			}
			res.data[rIt.next()] += mat.data[mIt.next()]*vec.data[vi];	
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
	public JavaTensor addvv(JavaTensor res, JavaTensor mat, JavaTensor vec1,
			JavaTensor vec2) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(vec1.size(), vec2.size());
		}
		JavaTensorIterator mIt = mat.iterator();
		JavaTensorIterator rIt = res.iterator();
		JavaTensorIterator v1It = vec1.iterator();
		JavaTensorIterator v2It = vec2.iterator();
		int v1i = v1It.next();
		while(mIt.hasNext()){
			if(!v2It.hasNext()){
				v2It = vec2.iterator();
				v1i = v1It.next();
			}
			res.data[rIt.next()] = mat.data[mIt.next()] + vec1.data[v1i]*vec2.data[v2It.next()];
		}
		return res;
	}


	@Override
	public JavaTensor addmv(JavaTensor res, final JavaTensor vec1, final JavaTensor mat,
			JavaTensor vec2) {
		if(res==null){
			res = factory.createTensor(mat.dims[0]);
		}
		JavaTensorIterator it = mat.iterator();
		JavaTensorIterator rIt = res.iterator();
		JavaTensorIterator aIt = vec1.iterator();
		while(it.hasNext()){
			float v = 0;
			JavaTensorIterator vIt = vec2.iterator();
			while(vIt.hasNext()){
				v+= mat.data[it.next()]*vec2.data[vIt.next()];
			}
			res.data[rIt.next()] = v + vec1.data[aIt.next()];
		}
		return res;
	}

	@Override
	public JavaTensor addmm(JavaTensor res, final JavaTensor mat, final JavaTensor mat1,
			JavaTensor mat2) {
		res = mm(res, mat1, mat2);
		return add(res, mat, res);
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
	public float sum(final JavaTensor tensor) {
		float sum = 0;
		JavaTensorIterator it = tensor.iterator();
		while(it.hasNext()){
			sum+= tensor.data[it.next()];
		}
		return sum;
	}

	@Override
	public float max(final JavaTensor tensor) {
		float max = Float.MIN_VALUE;
		JavaTensorIterator it = tensor.iterator();
		while(it.hasNext()){
			float val = tensor.data[it.next()];
			if(val > max)
				max = val;
		}
		return max;
	}

	@Override
	public float min(final JavaTensor tensor) {
		float min = Float.MAX_VALUE;
		JavaTensorIterator it = tensor.iterator();
		while(it.hasNext()){
			float val = tensor.data[it.next()];
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
	public int argmax(JavaTensor tensor) {
		float max = Float.MIN_VALUE;
		int index = -1;
		JavaTensorIterator it = tensor.iterator();
		int i = 0;
		while(it.hasNext()){
			float val = tensor.data[it.next()];
			if(val > max){
				max = val;
				index = i;
			}
			i++;
		}
		return index;
	}
	
	
	@Override
	public int argmin(final JavaTensor tensor) {
		float min = Float.MAX_VALUE;
		int index = -1;
		JavaTensorIterator it = tensor.iterator();
		int i = 0;
		while(it.hasNext()){
			float val = tensor.data[it.next()];
			if(val < min){
				min = val;
				index = i;
			}
			i++;
		}
		return index;
	}

}
