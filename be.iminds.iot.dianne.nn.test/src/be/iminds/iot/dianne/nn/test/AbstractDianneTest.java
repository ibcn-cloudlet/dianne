/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.test;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.osgi.framework.BundleContext;
import org.osgi.framework.FrameworkUtil;
import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import junit.framework.TestCase;

public class AbstractDianneTest extends TestCase {

    protected final BundleContext context = FrameworkUtil.getBundle(this.getClass()).getBundleContext();
    
    protected DiannePlatform platform;
    protected Dianne dianne;

    protected NeuralNetworkInstanceDTO nni = null;
	
    public void setUp() throws Exception {
    	ServiceReference rp =  context.getServiceReference(DiannePlatform.class.getName());
    	platform = (DiannePlatform) context.getService(rp);
    	
    	ServiceReference rd =  context.getServiceReference(Dianne.class.getName());
    	dianne = (Dianne) context.getService(rd);
    }
    
    public void tearDown(){
    	// tear down deployed NN modules after each test
    	if(nni != null)
    		platform.undeployNeuralNetwork(nni);
    }
    
    public void testServices() throws Exception {
    	Assert.assertNotNull(platform);
    	Assert.assertNotNull(dianne);
	}    
    
    protected NeuralNetwork deployNN(String name) throws Exception {
    	nni = platform.deployNeuralNetwork(name);
    	return dianne.getNeuralNetwork(nni).getValue();
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
    
    protected Module getModule(String id) throws Exception {
    	ServiceReference[] refs = context.getAllServiceReferences(Module.class.getName(), "(module.id="+id+")");
    	if(refs!=null){
	    	return (Module) context.getService(refs[0]);
    	}
    	return null;
    }
    
    protected List<Module> getModules() throws Exception {
    	List<Module> modules = new ArrayList<Module>();
    	ServiceReference[] refs = context.getAllServiceReferences(Module.class.getName(), null);
    	if(refs!=null){
	    	for(ServiceReference r : refs){
	    		modules.add((Module)context.getService(r));
	    	}
    	}
    	return modules;
    }
}
