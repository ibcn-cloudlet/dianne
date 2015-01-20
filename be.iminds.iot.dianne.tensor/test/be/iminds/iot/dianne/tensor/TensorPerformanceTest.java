package be.iminds.iot.dianne.tensor;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorMath;

public class TensorPerformanceTest<T extends Tensor<T>> {

	private TensorFactory<T> factory;
	private TensorMath<T> math;
	
	private int outSize = 1500;
	private int inSize = 28*28;
	
	private T parameters;
	private T gradParameters;
	private T weights;
	private T bias;
	private T gradWeights;
	private T gradBias;
	private T input;
	private T output;
	private T gradInput;
	private T gradOutput;
	
    @Before
    public void setUp() {
        factory = new TensorFactory(JavaTensor.class, JavaTensorMath.class);
        math = factory.getTensorMath();
        
		parameters = factory.createTensor(outSize, inSize+1);
		parameters.rand();
		weights = parameters.narrow(1, 0, inSize);
		bias = parameters.narrow(1, inSize, 1);

		gradParameters = factory.createTensor(outSize, inSize+1);		
		gradWeights = gradParameters.narrow(1, 0, inSize);
		gradBias = gradParameters.narrow(1, inSize, 1);

		input = factory.createTensor(inSize);
		input.rand();
		output = factory.createTensor(outSize);
		output.rand();
		
		gradInput = factory.createTensor(inSize);
		gradInput.rand();
		gradOutput = factory.createTensor(outSize);
		gradOutput.rand();
		
		
    }
	
	@Test
	public void testAddmv() {
		long t1 = System.currentTimeMillis();
		output = math.addmv(output, bias, weights, input);
		long t2 = System.currentTimeMillis();
		
		System.out.println("addmv: "+(t2-t1)+" ms");
		Assert.assertTrue((t2-t1)<5);
	}
	
	@Test
	public void testTmv(){
		long t1 = System.currentTimeMillis();
		gradInput = math.tmv(gradInput, weights, gradOutput);
		long t2 = System.currentTimeMillis();
		
		System.out.println("tmv: "+(t2-t1)+" ms");
		Assert.assertTrue((t2-t1)<5);
	}

	@Test
	public void testAddvv(){
		long t1 = System.currentTimeMillis();
		gradWeights = factory.getTensorMath().addvv(gradWeights, gradWeights, gradOutput, input);
		long t2 = System.currentTimeMillis();
		
		System.out.println("addvv: "+(t2-t1)+" ms");
		Assert.assertTrue((t2-t1)<5);
	}

}
