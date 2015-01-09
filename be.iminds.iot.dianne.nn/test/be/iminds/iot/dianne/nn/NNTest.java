package be.iminds.iot.dianne.nn;

import java.util.UUID;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.layer.Linear;

public class NNTest {

	@Test
	public void test() throws Exception {
		Sequential nn = new Sequential();
		nn.add(new Linear());
		nn.add(new Tanh());
		nn.add(new Linear());
		nn.add(new Tanh());
		
		UUID inputId = UUID.randomUUID();
		UUID outputId = UUID.randomUUID();

		nn.forward(inputId, null);
		
		Thread.sleep(1000);
		nn.backward(outputId, null);
		
		Thread.sleep(1000);
	}

}
