package be.iminds.iot.dianne.nn.util;

import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.util.DianneNeuralNetworkBuilder.Activation;

public class DianneNeuralNetworkBuilderTest {

	
	@Test
	public void testBuilder(){
		DianneNeuralNetworkBuilder builder = new DianneNeuralNetworkBuilder("Test");
		builder.addLinear(784, 20);
		builder.addReLU();
		builder.addLinear(20, 10);
		builder.addSoftmax();
		NeuralNetworkDTO nn = builder.create();
		System.out.println(DianneJSONConverter.toJsonString(nn, true));
	}
	
	@Test
	public void testMLP(){
		NeuralNetworkDTO nn = DianneNeuralNetworkBuilder.createMLP("MLP", 784, 10, Activation.Sigmoid, 100, 20);
		System.out.println(DianneJSONConverter.toJsonString(nn, true));
	}
}
