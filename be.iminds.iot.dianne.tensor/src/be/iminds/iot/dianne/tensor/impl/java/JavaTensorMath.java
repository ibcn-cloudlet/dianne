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
		for(int i=0;i<mat.dims[0];i++){
			float v = 0;
			for(int j=0;j<mat.dims[1];j++){
				v+= mat.get(i,j) * vec.get(j);
			}
			res.set(v, i);
		}
		return res;
	}
	
	@Override
	public JavaTensor tmv(JavaTensor res, final JavaTensor mat, final JavaTensor vec) {
		// TODO check dims?
		if(res==null){
			res = factory.createTensor(mat.dims[1]);
		}
		for(int i=0;i<mat.dims[1];i++){
			float v = 0;
			for(int j=0;j<mat.dims[0];j++){
				v+= mat.get(j,i) * vec.get(j);
			}
			res.set(v, i);
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
	public JavaTensor addmv(JavaTensor res, final JavaTensor vec1, final JavaTensor mat,
			JavaTensor vec2) {
		res = mv(res, mat, vec2);
		return add(res, vec1, res);
	}

	@Override
	public JavaTensor addmm(JavaTensor res, final JavaTensor mat, final JavaTensor mat1,
			JavaTensor mat2) {
		res = mm(res, mat1, mat2);
		return add(res, mat, res);
	}

	@Override
	public JavaTensor tanh(JavaTensor res, JavaTensor tensor) {
		Operator tanh = new Operator(){
			@Override
			public float apply(float... params) {
				return (float) Math.tanh(params[0]);
			}
		};
		return apply(res, tanh, tensor);
	}

	@Override
	public JavaTensor sigmoid(JavaTensor res, JavaTensor tensor) {
		Operator sigmoid = new Operator(){
			@Override
			public float apply(float... params) {
				return (float)(1.0/(1+Math.exp(-params[0])));
			}
		};
		return apply(res, sigmoid, tensor);
	}

	@Override
	public JavaTensor dtanh(JavaTensor res, JavaTensor tensor) {
		Operator dtanh = new Operator(){
			@Override
			public float apply(float... params) {
				return (1.0f - params[0]*params[0]);
			}
		};
		return apply(res, dtanh, tensor);
	}

	@Override
	public JavaTensor dsigmoid(JavaTensor res, JavaTensor tensor) {
		Operator dsigmoid = new Operator(){
			@Override
			public float apply(float... params) {
				return params[0]*(1.0f - params[0]);
			}
		};
		return apply(res, dsigmoid, tensor);
	}


}
