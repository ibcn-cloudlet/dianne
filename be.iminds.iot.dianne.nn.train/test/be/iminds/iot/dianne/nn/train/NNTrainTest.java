package be.iminds.iot.dianne.nn.train;

import junit.framework.Assert;

import org.junit.Test;

import be.iminds.iot.dianne.nn.train.dataset.MNISTDataset;
import be.iminds.iot.dianne.tensor.Tensor;

public class NNTrainTest {

	@Test
	public void testMNIST() {
		
		Dataset mnist = new MNISTDataset();
		
		Assert.assertEquals(60000, mnist.size());
		Assert.assertEquals(28*28, mnist.inputSize());
		Assert.assertEquals(10, mnist.outputSize());
		Tensor output = mnist.getOutputSample(0);

		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(0), 0.1f);
		Assert.assertEquals(1.0f, mnist.getOutputSample(0).get(5), 0.1f);
		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(9), 0.1f);
	}

}
