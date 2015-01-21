package be.iminds.iot.dianne.nn.train;

import java.util.ArrayList;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.container.Sequential;
import be.iminds.iot.dianne.nn.module.io.Input;
import be.iminds.iot.dianne.nn.module.io.Output;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.dataset.MNISTDataset;
import be.iminds.iot.dianne.nn.train.dataset.MNISTDataset.Set;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class NNTrainTest {

	private TensorFactory factory;
	
	@Before
	public void setUp(){
		factory = TensorFactory.getFactory(TensorFactory.TensorType.JAVA);
		

	}
	
	@Test
	public void testMNIST() {
		
		Dataset mnist = new MNISTDataset("/home/tverbele/MNIST/", Set.TRAIN);
		
		Assert.assertEquals(60000, mnist.size());
		Assert.assertEquals(28*28, mnist.inputSize());
		Assert.assertEquals(10, mnist.outputSize());
		Tensor output = mnist.getOutputSample(0);
		
		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(0), 0.1f);
		Assert.assertEquals(1.0f, mnist.getOutputSample(0).get(5), 0.1f);
		Assert.assertEquals(0.0f, mnist.getOutputSample(0).get(9), 0.1f);
		
		Dataset mnist2 = new MNISTDataset("/home/tverbele/MNIST/", Set.TEST);
		
		Assert.assertEquals(10000, mnist2.size());
		Assert.assertEquals(28*28, mnist2.inputSize());
		Assert.assertEquals(10, mnist2.outputSize());
	}

	@Test
	public void testEvaluation(){
		Tensor confusion = factory.createTensor(3,3);
		confusion.set(5.0f, 0, 0);
		confusion.set(3.0f, 0, 1);
		confusion.set(0.0f, 0, 2);
		confusion.set(2.0f, 1, 0);
		confusion.set(3.0f, 1, 1);
		confusion.set(1.0f, 1, 2);
		confusion.set(0.0f, 2, 0);
		confusion.set(2.0f, 2, 1);
		confusion.set(11.0f, 2, 2);
		
		Evaluation eval = new Evaluation(confusion);
		Assert.assertEquals(5.0f, eval.tp(0));
		Assert.assertEquals(2.0f, eval.fp(0));
		Assert.assertEquals(3.0f, eval.fn(0));
		Assert.assertEquals(17.0f, eval.tn(0));
		Assert.assertEquals(0.625f, eval.sensitivity(0), 0.000001f);
		Assert.assertEquals(0.894736842f, eval.specificity(0), 0.000001f);
		Assert.assertEquals(0.714285714f, eval.precision(0), 0.000001f);
		Assert.assertEquals(0.85f, eval.npv(0), 0.000001f);
		Assert.assertEquals(0.105263158f, eval.fallout(0), 0.000001f);
		Assert.assertEquals(0.285714286f, eval.fdr(0), 0.000001f);
		Assert.assertEquals(0.375f, eval.fnr(0), 0.000001f);
		Assert.assertEquals(0.814814815f, eval.accuracy(0), 0.000001f);
		Assert.assertEquals(0.185185185f, eval.error(0), 0.000001f);
		Assert.assertEquals(0.666666667, eval.f1(0), 0.000001f);
		Assert.assertEquals(0.541553391f, eval.mcc(0), 0.000001f);
		Assert.assertEquals(0.703703704f, eval.accuracy(), 0.000001f);
		Assert.assertEquals(0.296296296f, eval.error(), 0.000001f);
	}
	
	@Test
	public void testStochasticGradientTraining(){
		Dataset train = new MNISTDataset("/home/tverbele/MNIST/", Set.TRAIN);
		
		int noInput = train.inputSize();
		int noHidden = 20;
		int noOutput = train.outputSize();
		
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
		nn.add(new Sigmoid());
		nn.add(l2);
		nn.add(new Sigmoid());
		nn.add(out);
		
		
		Criterion loss = new MSECriterion();
		Trainer trainer = new StochasticGradient();
		trainer.train(in, out, modules, loss, train);
		
		// now evaluate
		System.out.println("Training done ... now evaluate...");
		Dataset test = new MNISTDataset("/home/tverbele/MNIST/", Set.TEST);
		Evaluator eval = new ArgMaxEvaluator();
		Evaluation result = eval.evaluate(in, out, test);
		
		System.out.println("Accuracy: "+result.accuracy());
		
		Assert.assertTrue(result.accuracy()>0.85);
	}
}
