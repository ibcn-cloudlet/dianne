package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath<JavaTensor> {

	private final JavaTensorFactory factory;
	
	JavaTensorMath(JavaTensorFactory factory) {
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
		for(int i=0;i<res.data.length;i++){
			res.data[i] = op.apply(t.data[i]);
		}
		return res;
	}

	private JavaTensor apply(JavaTensor res, final Operator op, final JavaTensor t1, final JavaTensor t2){
		// TODO check dims?
		if(res == null){
			res = factory.createTensor(t1.dims);
		}
		for(int i=0;i<res.data.length;i++){
			res.data[i] = op.apply(t1.data[i], t2.data[i]);
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
		float dot = 0;
		for(int i=0;i<vec1.data.length;i++){
			dot+= vec1.data[i] * vec2.data[i];
		}
		return dot;
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
		mv(res, mat, vec2);
		return add(res, vec1, res);
	}

	@Override
	public JavaTensor addmm(JavaTensor res, final JavaTensor mat, final JavaTensor mat1,
			JavaTensor mat2) {
		mm(res, mat1, mat2);
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
}
