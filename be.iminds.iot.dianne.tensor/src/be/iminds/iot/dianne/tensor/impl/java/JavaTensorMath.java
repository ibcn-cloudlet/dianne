package be.iminds.iot.dianne.tensor.impl.java;

import be.iminds.iot.dianne.tensor.TensorMath;

public class JavaTensorMath implements TensorMath<JavaTensor>{

	@Override
	public JavaTensor add(JavaTensor result, JavaTensor t1, JavaTensor t2) {
		if(result == null)
			result = new JavaTensor();
		
		System.out.println("Adding " + t1 + " with " + t2);
		
		return result;
	}

	@Override
	public JavaTensor mul(JavaTensor result, JavaTensor t1, JavaTensor t2) {
		if(result == null)
			result = new JavaTensor();
		
		System.out.println("Multiplying " + t1 + " with " + t2);
		
		return result;
	}

}
