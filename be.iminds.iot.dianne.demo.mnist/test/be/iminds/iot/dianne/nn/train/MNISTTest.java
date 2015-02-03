package be.iminds.iot.dianne.nn.train;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.demo.mnist.dataset.MNISTDataset;
import be.iminds.iot.dianne.demo.mnist.dataset.MNISTDataset.Set;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MNISTTest {

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
	
	@Test
	public void testMNISTTraining(){
		Dataset train = new MNISTDataset(factory, mnistDir, Set.TRAIN, false);
		
		int noInput = train.inputSize();
		int noHidden = 20;
		int noOutput = train.outputSize();
		
// TODO make this an OSGi based test?!		
		
//		Input in = new InputImpl(factory);
//		Output out = new OutputImpl(factory);
//		Linear l1 = new Linear(factory, noInput, noHidden);
//		Linear l2 = new Linear(factory, noHidden, noOutput);
//		ArrayList<Trainable> modules = new ArrayList<Trainable>();
//		modules.add(l1);
//		modules.add(l2);
//		
//		Sequential nn = new Sequential();
//		nn.add(in);
//		nn.add(l1);
//		nn.add(new Sigmoid(factory));
//		nn.add(l2);
//		nn.add(new Sigmoid(factory));
//		nn.add(out);
		
		
		Criterion loss = new MSECriterion(factory);
		Trainer trainer = new StochasticGradient();
//		trainer.train(in, out, modules, loss, train);
		
		// now evaluate
		System.out.println("Training done ... now evaluate...");
		Dataset test = new MNISTDataset(factory, mnistDir, Set.TEST, true);
		Evaluator eval = new ArgMaxEvaluator(factory);
//		Evaluation result = eval.evaluate(in, out, test);
//		
//		System.out.println("Accuracy: "+result.accuracy());
//		
//		Assert.assertTrue(result.accuracy()>0.85);
	}
}
