package be.iminds.iot.dianne.nn;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorFactory;

public class NNTest {

	private TensorFactory factory;
	
	@Before
	public void setUp(){
		factory = new JavaTensorFactory();
	}
	
	@Test
	public void test() throws Exception {
		Input in = new InputImpl(factory);
		Output out = new OutputImpl(factory);
		
		Sequential nn = new Sequential();
		nn.add(in);
		nn.add(new Linear(factory, 5,10));
		nn.add(new Tanh(factory));
		nn.add(new Linear(factory, 10,3));
		nn.add(new Tanh(factory));
		nn.add(out);
		
		Tensor input = factory.createTensor(5);
		input.fill(1.0f);
		in.input(input);
		
		Thread.sleep(1000);
		System.out.println("============");

		Tensor gradOutput = factory.createTensor(3);
		gradOutput.set(1.0f, 1);
		out.backpropagate(gradOutput);
		
		Thread.sleep(1000);
	}

}
