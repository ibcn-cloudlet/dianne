package be.iminds.iot.dianne.nn.train;

import junit.framework.Assert;

import org.junit.Test;

import be.iminds.iot.dianne.nn.train.dataset.MNISTDataset;

public class NNTrainTest {

	@Test
	public void testMNIST() {
		
		Dataset mnist = new MNISTDataset();
		
		Assert.assertEquals(60000, mnist.size());
		Assert.assertEquals(28*28, mnist.inputSize());
		Assert.assertEquals(1, mnist.outputSize());
		Assert.assertEquals(5.0f, mnist.getOutputSample(0).get(0), 0.1f);
	}

}
