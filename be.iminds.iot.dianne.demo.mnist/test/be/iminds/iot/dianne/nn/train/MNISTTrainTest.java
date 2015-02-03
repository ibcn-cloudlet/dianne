package be.iminds.iot.dianne.nn.train;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.tensor.TensorFactory;

//TODO make this an OSGi based test?!	

public class MNISTTrainTest {

	private TensorFactory factory;
	
	private final String mnistDir = "/home/tverbele/MNIST/";
	
	@Before
	public void setUp(){
        //factory = new JavaTensorFactory();
	}
	
	@Test
	public void testMNISTTraining(){
//		Dataset train = new MNISTDataset(factory, mnistDir, Set.TRAIN, false);
//		
//		int noInput = train.inputSize();
//		int noHidden = 20;
//		int noOutput = train.outputSize();
//		
//	
//		
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
//		
//		
//		Criterion loss = new MSECriterion(factory);
//		Trainer trainer = new StochasticGradient();
//		trainer.train(in, out, modules, loss, train);
//		
//		// now evaluate
//		System.out.println("Training done ... now evaluate...");
//		Dataset test = new MNISTDataset(factory, mnistDir, Set.TEST, true);
//		Evaluator eval = new ArgMaxEvaluator(factory);
//		Evaluation result = eval.evaluate(in, out, test);
//		
//		System.out.println("Accuracy: "+result.accuracy());
//		
//		Assert.assertTrue(result.accuracy()>0.85);
	}
}
