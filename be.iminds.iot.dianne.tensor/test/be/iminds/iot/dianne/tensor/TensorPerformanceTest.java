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
package be.iminds.iot.dianne.tensor;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;
import be.iminds.iot.dianne.tensor.impl.th.THTensorFactory;

@RunWith(Parameterized.class)
public class TensorPerformanceTest<T extends Tensor<T>> {

	private TensorFactory<T> factory;
	private TensorMath<T> math;
	private int count = 10;
	
	private int outSize = 1000;
	private int inSize = 231;
	private int kernelSize = 3;
	private int noInputPlanes = 3;
	private int noOutputPlanes = 100;
	
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
	
	private T inputnd;
	private T kernelsAndBias;
	private T kernels;
	private T biases;
	
	public TensorPerformanceTest(TensorFactory<T> f, String name) {
		this.factory = f;
	}

	@Parameters(name="{1}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { 
				{ new JavaTensorFactory(), "Java Tensor" },
				{ new THTensorFactory(), "TH Tensor" } 
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
		inputnd = factory.createTensor(noInputPlanes, inSize, inSize);
		inputnd.rand();
		output = factory.createTensor(outSize);
		output.rand();
		
		gradInput = factory.createTensor(inSize, inSize);
		gradInput.rand();
		gradOutput = factory.createTensor(outSize);
		gradOutput.rand();
		
		kernel = factory.createTensor(kernelSize, kernelSize);
		kernelsAndBias = factory.createTensor(noOutputPlanes*noInputPlanes*kernelSize*kernelSize+noOutputPlanes);
		kernelsAndBias.rand();
		
		kernels = kernelsAndBias.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelSize*kernelSize);
		kernels.reshape(noOutputPlanes, noInputPlanes, kernelSize, kernelSize);

		biases = kernelsAndBias.narrow(0, noOutputPlanes*noInputPlanes*kernelSize*kernelSize, noOutputPlanes);
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

//	@Test
//	public void testConv(){
//		long t1 = System.currentTimeMillis();
//		for(int i=0;i<count;i++)
//			factory.getTensorMath().convolution2D(null, input, kernel, 1, 1, 2, false);
//		long t2 = System.currentTimeMillis();
//		
//		float time = (float)(t2-t1)/count;
//		System.out.println("conv: "+time+" ms");
//		Assert.assertTrue((time)<5);
//	}
	
	@Test
	public void testSpatialConv(){
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			factory.getTensorMath().spatialconvolve(null, biases, inputnd, kernels, 5, 5, 0, 0);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("spatial conv: "+time+" ms");
		Assert.assertTrue((time)<5);
	}
	
	@Test
	public void testSpatialPooling(){
		long t1 = System.currentTimeMillis();
		for(int i=0;i<count;i++)
			factory.getTensorMath().spatialmaxpool(null, inputnd, 2, 2, 2, 2);
		long t2 = System.currentTimeMillis();
		
		float time = (float)(t2-t1)/count;
		System.out.println("spatial pool: "+time+" ms");
		Assert.assertTrue((time)<5);
	}
	
	// main method for visualvm profiling
	public static void main(String[] args) throws InterruptedException{
		TensorPerformanceTest<JavaTensor> test = new TensorPerformanceTest(new JavaTensorFactory(), "JavaTensor");
		test.setUp();
		test.testMv();
	}
}
