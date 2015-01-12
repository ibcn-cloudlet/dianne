package be.iminds.iot.dianne.nn.train;

import java.util.ArrayList;

import junit.framework.Assert;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.dataset.MNISTDataset;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
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

	@Test
	public void testStochasticGradientTraining(){
		Dataset data = new MNISTDataset();
		
		int noInput = data.inputSize();
		int noHidden = 1500;
		int noOutput = data.outputSize();
		
		Input in = new Input();
		Output out = new Output();
		Linear l1 = new Linear(noInput, noHidden);
		Linear l2 = new Linear(noHidden, noOutput);
		ArrayList<Trainable> modules = new ArrayList<Trainable>();
		modules.add(l1);
		modules.add(l2);
		
		Sequential nn = new Sequential();
		nn.add(in);
		nn.add(l1);
		nn.add(new Tanh());
		nn.add(l2);
		nn.add(new Tanh());
		nn.add(out);
		
		
		Criterion loss = new MSECriterion();
		Trainer trainer = new StochasticGradient();
		trainer.train(in, out, modules, loss, data);
		
	}
}
