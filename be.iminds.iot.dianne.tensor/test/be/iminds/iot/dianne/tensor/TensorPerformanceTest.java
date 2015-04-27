package be.iminds.iot.dianne.tensor;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.nd4j.ND4JTensor;
import be.iminds.iot.dianne.tensor.impl.nd4j.ND4JTensorFactory;

@RunWith(Parameterized.class)
public class TensorPerformanceTest<T extends Tensor<T>> {

	private TensorFactory<T> factory;
	private TensorMath<T> math;
	private int count = 10;
	
	private int outSize = 1000;
	private int inSize = 231;
	private int kernelSize = 11;
	
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
	private T kernel;
	
	public TensorPerformanceTest(TensorFactory<T> f, String name) {
		this.factory = f;
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
				{ new JavaTensorFactory(), "Java Tensor" },
				{ new ND4JTensorFactory(), "ND4J Tensor" } 
		});
	}

    @Before
    public void setUp() {
        math = factory.getTensorMath();
        
		parameters = factory.createTensor(outSize, inSize*inSize+1);
		parameters.rand();
		weights = parameters.narrow(1, 0, inSize*inSize);
		bias = parameters.narrow(1, inSize*inSize, 1);

		gradParameters = factory.createTensor(outSize, inSize*inSize+1);		
		gradWeights = gradParameters.narrow(1, 0, inSize*inSize);
		gradBias = gradParameters.narrow(1, inSize*inSize, 1);

		input = factory.createTensor(inSize, inSize);
		input.rand();
		output = factory.createTensor(outSize);
		output.rand();
		
		gradInput = factory.createTensor(inSize, inSize);
		gradInput.rand();
		gradOutput = factory.createTensor(outSize);
		gradOutput.rand();
		
		kernel = factory.createTensor(kernelSize, kernelSize);
				
    }
	
    @Test
    public void testMv(){
    	long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			output = math.mv(output, weights, input);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("mv: "+time+" ms");
		Assert.assertTrue((time)<5);
    }
    
	@Test
	public void testAddmv() {
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			output = math.addmv(output, bias, weights, input);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("addmv: "+time+" ms");
		Assert.assertTrue((time)<5);
	}
	
	@Test
	public void testTmv(){
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			gradInput = math.tmv(gradInput, weights, gradOutput);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("tmv: "+time+" ms");
		Assert.assertTrue((time)<5);
	}

	@Test
	public void testAddvv(){
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			gradWeights = factory.getTensorMath().addvv(gradWeights, gradWeights, gradOutput, input);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("addvv: "+time+" ms");
		Assert.assertTrue((time)<5);
	}

	@Test
	public void testConv(){
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			factory.getTensorMath().convolution2D(null, input, kernel, 1, 1, 2, false);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("conv: "+time+" ms");
		Assert.assertTrue((time)<5);
	}
	
	// main method for visualvm profiling
	public static void main(String[] args) throws InterruptedException{
		TensorPerformanceTest<ND4JTensor> test = new TensorPerformanceTest(new ND4JTensorFactory(), "ND4J");
		test.setUp();
		test.testMv();
	}
}
