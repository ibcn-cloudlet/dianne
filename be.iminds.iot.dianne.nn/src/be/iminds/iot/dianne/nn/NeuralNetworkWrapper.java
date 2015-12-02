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
package be.iminds.iot.dianne.nn;

import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.NeuralNetworkResult;
import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class NeuralNetworkWrapper implements NeuralNetwork {

	private int count = 0;

	private final DianneRepository repository;
	private final TensorFactory factory;
	
	private final NeuralNetworkInstanceDTO nn;
	private Map<UUID, Module> modules;
	private Map<UUID, Input> inputs;
	private Map<UUID, Output> outputs;
	private Map<UUID, Preprocessor> preprocessors;
	private Map<UUID, Trainable> trainables;
	
	private final BundleContext context;
	private ServiceRegistration<NeuralNetwork> nnReg;
	private ServiceRegistration<ForwardListener> forwardListenerReg;
	private ServiceRegistration<BackwardListener> backwardListenerReg;

	
	private Map<String, Deferred<NeuralNetworkResult>> inProgress = Collections.synchronizedMap(new HashMap<String, Deferred<NeuralNetworkResult>>());
	private Map<String, UUID> interestedModules = Collections.synchronizedMap(new HashMap<String, UUID>());
	
	private boolean valid = true;
	
	public NeuralNetworkWrapper(NeuralNetworkInstanceDTO nn, Collection<Module> modules, DianneRepository repo, TensorFactory factory, BundleContext context) {
		this.nn = nn;
		this.context = context;
		
		this.factory = factory;
		this.repository = repo;
		
		this.modules = modules.stream().collect(Collectors.toMap(m -> m.getId(), m -> m));
		this.inputs = modules.stream().filter(m -> m instanceof Input).map(i -> (Input)i).collect(Collectors.toMap(i -> i.getId(), i -> i));
		this.outputs = modules.stream().filter(m -> m instanceof Output).map(o -> (Output)o).collect(Collectors.toMap(o -> o.getId(), o -> o));
		this.preprocessors = modules.stream().filter(m -> m instanceof Preprocessor).map(p -> (Preprocessor)p).collect(Collectors.toMap(p -> p.getId(), p -> p));
		this.trainables = modules.stream().filter(m -> m instanceof Trainable).map(t -> (Trainable)t).collect(Collectors.toMap(t -> t.getId(), t -> t));

	}
	
	@Override
	public NeuralNetworkInstanceDTO getNeuralNetworkInstance() {
		return nn;
	}

	@Override
	public Promise<NeuralNetworkResult> forward(UUID inputId, UUID outputId, Tensor in, String... tags){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Input input = null;
		if(inputId!=null){
			input = inputs.get(inputId);
		}
		if(input==null){
			input = inputs.values().iterator().next();
		}
		
		String tag = getTag();
		if(outputId!=null){
			interestedModules.put(tag, outputId);
		}
		
		Deferred<NeuralNetworkResult> d = new Deferred<>();
		inProgress.put(tag, d);
		
		input.input(in, addTag(tags, tag));
		
		return d.getPromise();
	}
	
	@Override
	public Promise<NeuralNetworkResult> backward(UUID outputId, UUID inputId, Tensor gradOut, String... tags){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Output output = null;
		if(outputId!=null){
			output = outputs.get(outputId);
		}
		if(output==null){
			output = outputs.values().iterator().next();
		}
		
		String tag = getTag();
		if(outputId!=null){
			interestedModules.put(tag, outputId);
		}
		
		Deferred<NeuralNetworkResult> d = new Deferred<>();
		inProgress.put(tag, d);
		
		output.backpropagate(gradOut, addTag(tags, tag));
		
		return d.getPromise();
	}
	
	

	private String getTag(){
		synchronized(this){
			return ""+count++;
		}
	}
	
	
	
	private String[] addTag(String[] tags, String tag){
		if(tag==null)
			return tags;
		
		int l = tags.length;
		String[] t = new String[l+1];
		System.arraycopy(tags, 0, t, 1, tags.length);
		t[0] = tag;
		return t;
	}
	
	private String[] removeTag(String[] tags, String tag){
		int l = tags.length;
		if(l <= 1){
			return new String[0];
		}
		String[] t = new String[l-1];
		int i = 0;
		for(String s : tags){
			if(!s.equals(tag)){
				t[i++] = s;
			}
		}
		return t;
	}
	
	
	void register(){
		Dictionary<String, Object> propertiesFw = new Hashtable<String, Object>();
		propertiesFw.put("targets", new String[]{nn.id.toString()});
		propertiesFw.put("aiolos.unique", true);
		forwardListenerReg = context.registerService(ForwardListener.class, new ForwardListener() {
			
			@Override
			public void onForward(UUID moduleId, Tensor output, String... tags) {
				if(tags==null || tags.length==0)
					return;
				
				String tag = tags[0];
				
				if(interestedModules.containsKey(tag)){
					if(!moduleId.equals(interestedModules.get(tag))){
						return;
					}
				}
				
				interestedModules.remove(tag);
				Deferred<NeuralNetworkResult> d = inProgress.remove(tag);
				if(d!=null){
					NeuralNetworkResult r = new NeuralNetworkResult(output, removeTag(tags, tag));
					d.resolve(r);
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				if(tags==null || tags.length==0)
					return;
				
				String tag = tags[0];
				
				if(interestedModules.containsKey(tag)){
					if(!moduleId.equals(interestedModules.get(tag))){
						return;
					}
				}
				
				interestedModules.remove(tag);
				Deferred<NeuralNetworkResult> d = inProgress.remove(tag);
				if(d!=null){
					d.fail(e);
				}
			}
		}, propertiesFw);
	
		Dictionary<String, Object> propertiesBw = new Hashtable<String, Object>();
		propertiesBw.put("targets", new String[]{nn.id.toString()});
		propertiesBw.put("aiolos.unique", true);
		backwardListenerReg = context.registerService(BackwardListener.class, new BackwardListener() {
			
			@Override
			public void onBackward(UUID moduleId, Tensor gradInput, String... tags) {
				if(tags==null || tags.length==0) {
					return;
				}
				
				String tag = tags[0];
				
				if(interestedModules.containsKey(tag)){
					if(!moduleId.equals(interestedModules.get(tag))){
						return;
					}
				}
				
				interestedModules.remove(tag);
				Deferred<NeuralNetworkResult> d = inProgress.remove(tag);
				if(d!=null){
					NeuralNetworkResult r = new NeuralNetworkResult(gradInput, removeTag(tags, tag));
					d.resolve(r);
				} else {
					System.err.println("No deferred for tag "+tag+" ?!");
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				if(tags==null || tags.length==0) {
					return;
				}
				
				String tag = tags[0];
				
				if(interestedModules.containsKey(tag)){
					if(!moduleId.equals(interestedModules.get(tag))){
						return;
					}
				}
				
				interestedModules.remove(tag);
				Deferred<NeuralNetworkResult> d = inProgress.remove(tag);
				if(d!=null){
					d.fail(e);
				} else {
					System.err.println("No deferred for tag "+tag+" ?!");
				}
			}
		}, propertiesBw);	
		
		Dictionary<String, Object> properties = new Hashtable<String, Object>();
		properties.put("nn.id", nn.id.toString());
		properties.put("aiolos.export", false);
		nnReg = context.registerService(NeuralNetwork.class, this, properties);
		
		valid = true;
	}
	
	void unregister(){
		valid = false;
		
		forwardListenerReg.unregister();
		backwardListenerReg.unregister();
		nnReg.unregister();
	}

	@Override
	public Input getInput(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		if(inputs.size() > 1){
			throw new RuntimeException("This neural network has more than one input");
		}
		return inputs.values().iterator().next();
	}
	
	@Override
	public Map<UUID, Input> getInputs() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return inputs;
	}

	@Override
	public Output getOutput(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		return outputs.values().iterator().next();
	}
	
	@Override
	public Map<UUID, Output> getOutputs() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return outputs;
	}

	@Override
	public String[] getOutputLabels() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		return outputs.values().iterator().next().getOutputLabels();
	}

	@Override
	public String[] getOutputLabels(UUID outputId) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Output output = outputs.get(outputId);
		if(output==null){
			throw new RuntimeException("This neural network does not have output "+outputId);
		}
		return output.getOutputLabels();
	}

	@Override
	public Map<UUID, Trainable> getTrainables() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return trainables;
	}

	@Override
	public Map<UUID, Preprocessor> getPreprocessors() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return preprocessors;
	}
	
	@Override
	public Map<UUID, Module> getModules() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return modules;
	}

	@Override
	public void setParameters(Map<UUID, Tensor> parameters) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		parameters.entrySet().forEach(e -> setParameters(e.getKey(), e.getValue()));
	}

	@Override
	public void setParameters(UUID moduleId, Tensor parameters) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Trainable t = trainables.get(moduleId);
		if(t!=null){
			t.setParameters(parameters);
		} else {
			Preprocessor p = preprocessors.get(moduleId);
			if(p!=null){
				p.setParameters(parameters);
			}
		}
	}

	@Override
	public Map<UUID, Tensor> getParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Map<UUID, Tensor> parameters = trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().getParameters()));
		preprocessors.entrySet().stream().forEach(e -> parameters.put(e.getKey(), e.getValue().getParameters()));
		return parameters;
	}
	
	@Override
	public void randomizeParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		trainables.values().stream().forEach(t -> t.randomize());
	}

	@Override
	public void storeParameters(String... tag) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		repository.storeParameters(nn.id, getParameters(), tag);
	}

	@Override
	public void storeDeltaParameters(Map<UUID, Tensor> previous, String... tag) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Map<UUID, Tensor> deltaParameters = trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> factory.getTensorMath().sub(null,
						e.getValue().getParameters(), previous.get(e.getKey()))));
		repository.accParameters(nn.id, deltaParameters, tag);
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String... tag) throws Exception {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		Map<UUID, Tensor> parameters = repository.loadParameters(nn.name, tag);
		setParameters(parameters);
		return parameters;
	}
}
