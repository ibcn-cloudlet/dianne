package be.iminds.iot.dianne.nn.test;

import org.junit.Assert;
import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetRangeAdapter;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.train.api.Criterion;
import be.iminds.iot.dianne.api.nn.train.api.Evaluation;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;

public class DianneTest extends AbstractDianneTest {

    private Dataset mnist;
    
    public void setUp() throws Exception {
    	super.setUp();
    	
    	ServiceReference[] rds = context.getAllServiceReferences(Dataset.class.getName(), null);
    	for(ServiceReference rd : rds){
    		Dataset d = (Dataset) context.getService(rd);
    		if(d.getName().equals("MNIST")){
    			mnist = d;
    		}
    	}
    }
    
    public void testLinearSigmoid() throws Exception {
    	sgd("test-mnist-linear-sigmoid", 10, 1, 0.05f, new MSECriterion(factory));
    }
    
    public void testLinearSoftmax() throws Exception {
    	sgd("test-mnist-linear-softmax", 10, 1, 0.001f, new NLLCriterion(factory));
    }

    public void testLinearSigmoidNorm() throws Exception {
    	sgd("test-mnist-linear-sigmoid-norm", 10, 1, 0.05f, new MSECriterion(factory));
    }
    
    public void testLinearSoftmaxNorm() throws Exception {
    	sgd("test-mnist-linear-softmax-norm", 10, 1, 0.001f, new NLLCriterion(factory));
    }
    
    public void testConvReLUNorm() throws Exception {
    	sgd("test-mnist-conv-relu-norm", 10, 2, 0.0001f, new NLLCriterion(factory));
    }

    public void testConvTanhNorm() throws Exception {
    	sgd("test-mnist-conv-tanh-norm", 10, 2, 0.0001f, new NLLCriterion(factory));
    }
    
    private void sgd(String config, int batch, int epochs, float learningRate, Criterion loss) throws Exception {
    	modules = deployNN("nn/"+config+".txt");
    	
    	StochasticGradient trainer = new StochasticGradient(batch, epochs, learningRate, 0.0f);
    	
    	Input input = getInput();
    	Output output = getOutput();
    	
    	Dataset train = new DatasetRangeAdapter(mnist, 0, 60000);
    	trainer.train(input, output, getTrainable(), getPreprocessors(), loss, train);
    	
    	ArgMaxEvaluator evaluator = new ArgMaxEvaluator(factory);
    	Dataset test = new DatasetRangeAdapter(mnist, 60000, 70000);
    	
    	Evaluation eval = evaluator.evaluate(input, output, test);
    	System.out.println("Accuracy "+eval.accuracy());
    	Assert.assertTrue(eval.accuracy()>0.6);
    }

}
