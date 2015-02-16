package be.iminds.iot.dianne.nn.test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Dictionary;
import java.util.List;
import java.util.UUID;

import junit.framework.TestCase;

import org.junit.Assert;
import org.osgi.framework.BundleContext;
import org.osgi.framework.FrameworkUtil;
import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.DatasetAdapter;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.nn.runtime.util.DianneJSONParser;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class DianneTest extends TestCase {

    private final BundleContext context = FrameworkUtil.getBundle(this.getClass()).getBundleContext();
    
    private TensorFactory factory;
    private Dataset mnist;
    
    public void setUp(){
    	ServiceReference rf = context.getServiceReference(TensorFactory.class.getName());
    	factory = (TensorFactory) context.getService(rf);
    	
    	ServiceReference rd = context.getServiceReference(Dataset.class.getName());
    	mnist = (Dataset) context.getService(rd);
    }
    
    public void testLinear() throws Exception {
    	ServiceReference ref =  context.getServiceReference(ModuleManager.class.getName());
    	Assert.assertNotNull(ref);
    	
    	ModuleManager mm = (ModuleManager) context.getService(ref);
    	
    	String json = new String(Files.readAllBytes(Paths.get("nn/test-mnist-linear.txt")));
    	
    	System.out.println(json);
    	List<Dictionary<String, Object>> configs = DianneJSONParser.parseJSON(json);
    	
    	for(Dictionary<String, Object> config : configs){
    		mm.deployModule(config);
    	}
    	
    	ServiceReference ri =  context.getServiceReference(Input.class.getName());
    	Assert.assertNotNull(ri);
    	Input input = (Input) context.getService(ri);
    	
    	ServiceReference ro =  context.getServiceReference(Output.class.getName());
    	Assert.assertNotNull(ro);
    	Output output = (Output) context.getService(ro);
    	
    	int batch = 10;
    	int epochs = 1;
    	StochasticGradient trainer = new StochasticGradient(batch, epochs);
    	Criterion loss = new MSECriterion(factory);
    	
    	List<Trainable> modules = new ArrayList<Trainable>();
    	ServiceReference[] refs = context.getAllServiceReferences(Trainable.class.getName(), null);
    	for(ServiceReference r : refs){
    		modules.add((Trainable)context.getService(r));
    	}
    	
    	Dataset train = new DatasetAdapter(mnist, 0, 60000);
    	trainer.train(input, output, modules, loss, train);

    	ArgMaxEvaluator evaluator = new ArgMaxEvaluator(factory);
    	Dataset test = new DatasetAdapter(mnist, 60000, 70000);
    	
    	Evaluation eval = evaluator.evaluate(input, output, test);
    	System.out.println("Accuracy "+eval.accuracy());
    	Assert.assertTrue(eval.accuracy()>0.8);
    	
    	for(Dictionary<String, Object> config : configs){
    		String id = (String) config.get("module.id");
    		mm.undeployModule(UUID.fromString(id));
    	}
    }
}
