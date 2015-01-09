package be.iminds.iot.dianne.nn;

import java.util.UUID;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.module.layer.Linear;

public class NNTest {

	@Test
	public void test() throws Exception {
		Input in = new Input();
		Output out = new Output();
		
		Sequential nn = new Sequential();
		nn.add(in);
		nn.add(new Linear());
		nn.add(new Tanh());
		nn.add(new Linear());
		nn.add(new Tanh());
		nn.add(out);
		
		in.input(null);
		
		Thread.sleep(1000);
		System.out.println("============");
		
		out.expected(null);
		
		Thread.sleep(1000);
	}

}
