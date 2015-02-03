package be.iminds.iot.dianne.nn.train;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.mnist.MNISTDataset;
import be.iminds.iot.dianne.dataset.mnist.MNISTDataset.Set;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MNISTReadTest {

	private TensorFactory factory;
	
	private final String mnistDir = "/home/tverbele/MNIST/";
	
	@Before
	public void setUp(){
        //factory = new JavaTensorFactory();
	}
	
	@Test
	public void testReadMNIST() {
		
		Dataset mnist = new MNISTDataset(factory, mnistDir, Set.TRAIN);
		
		Assert.assertEquals(60000, mnist.size());
		Assert.assertEquals(28*28, mnist.inputSize());
		Assert.assertEquals(10, mnist.outputSize());
		Tensor output = mnist.getOutputSample(0);
		
		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(0), 0.1f);
		Assert.assertEquals(1.0f, mnist.getOutputSample(0).get(5), 0.1f);
		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(9), 0.1f);
		
		Dataset mnist2 = new MNISTDataset(factory, mnistDir, Set.TEST);
		
		Assert.assertEquals(10000, mnist2.size());
		Assert.assertEquals(28*28, mnist2.inputSize());
		Assert.assertEquals(10, mnist2.outputSize());
	}
}
