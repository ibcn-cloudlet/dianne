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

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.nn.runtime.util.DianneJSONParser;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class AbstractDianneTest extends TestCase {

    protected final BundleContext context = FrameworkUtil.getBundle(this.getClass()).getBundleContext();
    
    protected TensorFactory factory;
    protected ModuleManager mm;

    protected List<UUID> modules = null;
	
    public void setUp() throws Exception {
       	ServiceReference rf = context.getServiceReference(TensorFactory.class.getName());
    	factory = (TensorFactory) context.getService(rf);
    	
    	ServiceReference rmm =  context.getServiceReference(ModuleManager.class.getName());
    	mm = (ModuleManager) context.getService(rmm);
    }
    
    public void tearDown(){
    	// tear down deployed NN modules after each test
    	if(modules!=null){
    		undeployNN(modules);
    		modules = null;
    	}
    }
    
    protected List<UUID> deployNN(String configLocation) throws Exception {
    	String json = new String(Files.readAllBytes(Paths.get(configLocation)));
    	List<Dictionary<String, Object>> configs = DianneJSONParser.parseJSON(json);
    	
    	List<UUID> ids = new ArrayList<UUID>();
    	for(Dictionary<String, Object> config : configs){
    		try {
	    		mm.deployModule(config);
	    		
	    		String id = (String)config.get("module.id");
	    		ids.add(UUID.fromString(id));
    		} catch(InstantiationException e){}
    	}
    	
    	return ids;
    }
    
    protected void undeployNN(List<UUID> modules){
    	for(UUID id : modules){
    		mm.undeployModule(id);
    	}
    }
    
    protected Input getInput(){
    	ServiceReference ri =  context.getServiceReference(Input.class.getName());
    	Assert.assertNotNull(ri);
    	Input input = (Input) context.getService(ri);
    	return input;
    }
    
    protected Output getOutput(){
    	ServiceReference ro =  context.getServiceReference(Output.class.getName());
    	Assert.assertNotNull(ro);
    	Output output = (Output) context.getService(ro);
    	return output;
    }
    
    protected List<Trainable> getTrainable() throws Exception {
    	List<Trainable> modules = new ArrayList<Trainable>();
    	ServiceReference[] refs = context.getAllServiceReferences(Trainable.class.getName(), null);
    	for(ServiceReference r : refs){
    		modules.add((Trainable)context.getService(r));
    	}
    	return modules;
    }
    
    protected List<Preprocessor> getPreprocessors() throws Exception {
    	List<Preprocessor> modules = new ArrayList<Preprocessor>();
    	ServiceReference[] refs = context.getAllServiceReferences(Preprocessor.class.getName(), null);
    	if(refs!=null){
	    	for(ServiceReference r : refs){
	    		modules.add((Preprocessor)context.getService(r));
	    	}
    	}
    	return modules;
    }
}
