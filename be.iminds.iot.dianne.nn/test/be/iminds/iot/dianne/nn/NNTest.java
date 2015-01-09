package be.iminds.iot.dianne.nn;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class NNTest {

	private TensorFactory factory;
	
	@Before
	public void setUp(){
		factory = TensorFactory.getFactory(TensorFactory.TensorType.JAVA);

	}
	
	@Test
	public void test() throws Exception {
		Input in = new Input();
		Output out = new Output();
		
		Sequential nn = new Sequential();
		nn.add(in);
		nn.add(new Linear(5,10));
		nn.add(new Tanh());
		nn.add(new Linear(10,3));
		nn.add(new Tanh());
		nn.add(out);
		
		Tensor input = factory.createTensor(5);
		input.fill(1.0f);
		in.input(input);
		
		Thread.sleep(1000);
		System.out.println("============");

		Tensor expected = factory.createTensor(3);
		expected.set(1.0f, 1);
		out.expected(expected);
		
		Thread.sleep(1000);
	}

}
