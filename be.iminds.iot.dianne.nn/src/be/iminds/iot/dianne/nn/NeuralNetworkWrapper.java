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

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.NeuralNetworkResult;
import be.iminds.iot.dianne.api.nn.NeuralNetworkSequenceResult;
import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Memory;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class NeuralNetworkWrapper implements NeuralNetwork {

	private int count = 0;

	private final DianneRepository repository;

	private final NeuralNetworkInstanceDTO nn;
	private Map<UUID, Module> modules;
	private Map<UUID, Input> inputs;
	private Map<UUID, Output> outputs;
	private Map<UUID, Preprocessor> preprocessors;
	private Map<UUID, Memory> memories;
	private Map<UUID, Trainable> trainables;
	
	private final BundleContext context;
	private ServiceRegistration<NeuralNetwork> nnReg;
	private ServiceRegistration<ForwardListener> forwardListenerReg;
	private ServiceRegistration<BackwardListener> backwardListenerReg;

	private Map<String, Progress> inProgress = Collections.synchronizedMap(new HashMap<String, Progress>());
	private Map<String, List<UUID>> interestedModules = Collections.synchronizedMap(new HashMap<String, List<UUID>>());

	private Map<UUID, List<Tensor>> sequenceInputs = new HashMap<>();
	private Map<UUID, List<Tensor>> sequenceOutputs = new HashMap<>();
	private Map<UUID, List<Tensor>> sequenceMemories = new HashMap<>();
	private Map<UUID, List<Tensor>> sequenceGradInputs = new HashMap<>();

	
	private boolean valid = true;
	
	public NeuralNetworkWrapper(NeuralNetworkInstanceDTO nn, Collection<Module> modules, DianneRepository repo, BundleContext context) {
		this.nn = nn;
		this.context = context;
		
		this.repository = repo;
		
		this.modules = modules.stream().collect(Collectors.toMap(m -> m.getId(), m -> m));
		this.inputs = modules.stream().filter(m -> m instanceof Input).map(i -> (Input)i).collect(Collectors.toMap(i -> i.getId(), i -> i));
		this.outputs = modules.stream().filter(m -> m instanceof Output).map(o -> (Output)o).collect(Collectors.toMap(o -> o.getId(), o -> o));
		this.preprocessors = modules.stream().filter(m -> m instanceof Preprocessor).map(p -> (Preprocessor)p).collect(Collectors.toMap(p -> p.getId(), p -> p));
		this.memories = modules.stream().filter(m -> m instanceof Memory).map(m -> (Memory)m).collect(Collectors.toMap(m -> m.getId(), m -> m));
		this.trainables = modules.stream().filter(m -> m instanceof Trainable).map(t -> (Trainable)t).collect(Collectors.toMap(t -> t.getId(), t -> t));

	}

	@Override
	public UUID getId() {
		return nn.id;
	}
	
	@Override
	public NeuralNetworkInstanceDTO getNeuralNetworkInstance() {
		return nn;
	}

	@Override
	public UUID getModuleId(String name) {
		try {
			return getNeuralNetworkInstance().modules.entrySet().stream().filter(e -> name.equalsIgnoreCase(e.getValue().module.properties.get("name"))).findFirst().map(e -> e.getKey()).get();
		} catch(NoSuchElementException e){
			throw new RuntimeException("Module with name "+name+" not found");
		}
	}

	@Override
	public UUID[] getModuleIds(String... names) {
		UUID[] result = new UUID[names.length];
		for(int i=0;i<names.length;i++){
			result[i] = getModuleId(names[i]);
		}
		return result;
	}
	
	@Override
	public Promise<NeuralNetworkResult> forward(UUID[] inputIds, UUID[] outputIds, Tensor[] ins, String... tags){
		if(!valid)
			throw new RuntimeException("This neural network object is no longer valid");
		
		String tag = getTag();
		String[] t = addTag(tags, tag);
		
		// first trigger all memories
		memories.values().forEach(m -> m.triggerForward(t));
		
		if(outputIds!=null) {
			for(UUID outputId : outputIds)
				addInterest(outputId, tag);
		} else {
			// just mark all outputs as interested
			nn.modules.values().stream().filter(m -> m.module.type.equals("Output")).map(m -> m.moduleId).forEach(id -> addInterest(id, tag));
		}
		
		Progress p = new Progress(outputIds);
		inProgress.put(tag, p);
		
		
		if(inputIds == null){
			Input input = inputs.values().iterator().next();
			input.input(ins[0], t);
		} else {
			for(int i=0;i<ins.length;i++){
				Input input = inputs.get(inputIds[i]);
				input.input(ins[i], t);
			}
		}
		
		return p.getPromise();
	}
	
	@Override
	public Promise<NeuralNetworkResult> forward(UUID inputId, UUID outputId, Tensor in, String... tags){
		return forward(inputId == null ? null : new UUID[]{inputId}, outputId == null ? null : new UUID[]{outputId}, new Tensor[]{in}, tags);	
	}
	
	@Override
	public Tensor forward(Tensor input, String... tags){
		Tensor result = null;
		Promise<NeuralNetworkResult> p = forward(null, null, input, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error forwarding input", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error forwarding input", e);
		}
		return result;
	}
	
	
	public Promise<NeuralNetworkSequenceResult> forward(UUID inputId, UUID outputId, List<Tensor> input, String... tags){
		return forward(inputId == null ? null : new UUID[]{inputId}, outputId == null ? null : new UUID[]{outputId}, new List[]{input}, tags);
	}
	
	public Promise<NeuralNetworkSequenceResult> forward(UUID[] inputIds, UUID[] outputIds, List<Tensor>[] inputs, String... tags){
		if(!valid)
			throw new RuntimeException("This neural network object is no longer valid");
		
		// store inputs
		if(inputIds == null){
			sequenceInputs.put(null, inputs[0]);
		} else {
			for(int i=0;i<inputIds.length;i++){
				sequenceInputs.put(inputIds[i], inputs[i]);
			}
		}

		// TODO what if other sequence already executing?!
		return forward(0, inputIds, outputIds, inputs, tags);
	}

	private Promise<NeuralNetworkSequenceResult> forward(int index, UUID[] inputIds, UUID[] outputIds, List<Tensor>[] inputs, String... tags){
		int sequenceLength = inputs[0].size();

		storeSequenceMemories(index);

		return forward(inputIds, outputIds, getSequenceInputs(index, inputs), tags).then(p -> {
			// store output
			storeSequenceOutputs(index, p.getValue().tensors);

			// increment counter
			int next = index + 1;
			
			// check if we are done
			if(next == sequenceLength){
				Deferred<NeuralNetworkSequenceResult> d = new Deferred<>();
				d.resolve(new NeuralNetworkSequenceResult(sequenceOutputs, tags));
				return d.getPromise();
			} else {
				return forward(next, inputIds, outputIds, inputs, tags);
			}
		});
	}
	
	public List<Tensor> forward(List<Tensor> input, String... tags){
		List<Tensor> result = null;
		Promise<NeuralNetworkSequenceResult> p = forward(null, null, input, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error forwarding input", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error forwarding input", e);
		}
		return result;
	}
	
	
	@Override
	public Promise<NeuralNetworkResult> backward(UUID[] outputIds, UUID[] inputIds, Tensor[] gradOuts, boolean accGradParameters, String... tags){
		if(!valid)
			throw new RuntimeException("This neural network object is no longer valid");
		
		String tag = getTag();
		String[] t = addTag(tags, tag);

		// first trigger all memories
		memories.values().forEach(m -> m.triggerBackward(t));
		
		if(inputIds!=null) {
			for(UUID inputId : inputIds){
				addInterest(inputId, tag);
			}
		} else {
			// just mark all inputs as interested
			nn.modules.values().stream().filter(m -> m.module.type.equals("Input")).map(m -> m.moduleId).forEach(id -> addInterest(id, tag));
		}
		
		Progress p = new Progress(inputIds);
		inProgress.put(tag, p);
		
		if(outputIds==null){
			Output output = outputs.values().iterator().next();
			output.backpropagate(gradOuts[0], t);
		} else {
			for(int i=0;i<gradOuts.length;i++){
				Output output = outputs.get(outputIds[i]);
				output.backpropagate(gradOuts[i], t);
			}
		}
		
		if(accGradParameters){
			return p.getPromise().then(		
				pp -> {	
					// Accumulate gradient weights
					getTrainables().values().stream().forEach(Trainable::accGradParameters);
					
					return pp;
				});
		} else {
			return p.getPromise();
		}
	}

	@Override
	public Promise<NeuralNetworkResult> backward(UUID outputId, UUID inputId, Tensor gradOut, boolean accGradParameters, String... tags){
		return backward(outputId == null ? null : new UUID[]{outputId}, inputId == null? null : new UUID[]{inputId}, new Tensor[]{gradOut}, accGradParameters, tags);
	}
	
	@Override
	public Tensor backward(Tensor gradOutput, boolean accGradParameters, String... tags){
		Tensor result = null;
		Promise<NeuralNetworkResult> p = backward(null, null, gradOutput, accGradParameters, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error back propagating gradOutput", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error back propagating gradOutput", e);
		}
		return result;
	}
	
	
	public Promise<NeuralNetworkSequenceResult> backward(UUID outputId, UUID inputId, List<Tensor> gradOutput, boolean accGradParameters, String... tags){
		return backward(outputId == null ? null : new UUID[]{outputId}, inputId == null ? null : new UUID[]{inputId}, new List[]{gradOutput}, accGradParameters, tags);
	}
	
	public Promise<NeuralNetworkSequenceResult> backward(UUID[] outputIds, UUID[] inputIds, List<Tensor>[] gradOutputs, boolean accGradParameters, String... tags){
		if(!valid)
			throw new RuntimeException("This neural network object is no longer valid");
		
	
		// TODO what if other sequence already executing?!
		return backward(gradOutputs[0].size()-1, outputIds, inputIds, gradOutputs, accGradParameters, tags);
	}

	private Promise<NeuralNetworkSequenceResult> backward(int index, UUID[] outputIds, UUID[] inputIds, List<Tensor>[] gradOutputs, boolean accGradParameters, String... tags){
		// first forward again with correct input and memories
		loadSequenceMemories(index);
		
		Tensor[] ins = new Tensor[inputIds == null ? 1 : inputIds.length];
		if(inputIds == null){
			ins[0] = sequenceInputs.values().iterator().next().get(index);
		} else {
			for(int i=0;i<inputIds.length;i++){
				ins[i] = sequenceInputs.get(inputIds[i]).get(index);
			}
		}
		
		return forward(inputIds, outputIds, ins, tags).then(p -> {
			return backward(outputIds, inputIds, getSequenceInputs(index, gradOutputs), accGradParameters, tags).then( pp -> {
				
				storeSequenceGradInputs(index, pp.getValue().tensors);

				// decrement counter
				int next = index - 1;

				// check if we are done
				if(next < 0){
					Deferred<NeuralNetworkSequenceResult> d = new Deferred<>();
					d.resolve(new NeuralNetworkSequenceResult(sequenceGradInputs, tags));
					return d.getPromise();
				} else {
					return backward(next, outputIds, inputIds, gradOutputs, accGradParameters, tags);
				}
				
			});
		});
	}
	

	public List<Tensor> backward(List<Tensor> gradOutput, boolean accGradParameters, String... tags){
		List<Tensor> result = null;
		Promise<NeuralNetworkSequenceResult> p = backward(null, null, gradOutput, accGradParameters, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error back propagating gradOutput", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error back propagating gradOutput", e);
		}
		return result;
	}
	
	
	private void storeSequenceOutputs(int index, Map<UUID, Tensor> outputs){
		outputs.entrySet().forEach(e -> {
			List<Tensor> outs = sequenceOutputs.get(e.getKey());
			if(outs == null){
				outs = new ArrayList<>();
				sequenceOutputs.put(e.getKey(), outs);
			}
			
			while(outs.size() <= index){
				outs.add(new Tensor());
			}
			
			e.getValue().copyInto(outs.get(index));
		});
	}
	
	private void storeSequenceMemories(int index){
		memories.entrySet().forEach(e ->{
			List<Tensor> mems = sequenceMemories.get(e.getKey());
			if(mems == null){
				mems = new ArrayList<Tensor>();
				sequenceMemories.put(e.getKey(), mems);
			}
			
			while(mems.size() <= index){
				mems.add(new Tensor());
			}
			
			e.getValue().getMemory().copyInto(mems.get(index));
		});	
	}
	
	private void storeSequenceGradInputs(int index, Map<UUID, Tensor> gradInputs){
		gradInputs.entrySet().forEach(e -> {
			List<Tensor> gradIns = sequenceGradInputs.get(e.getKey());
			if(gradIns == null){
				gradIns = new ArrayList<>();
				sequenceGradInputs.put(e.getKey(), gradIns);
			}
			
			while(gradIns.size() <= index){
				gradIns.add(new Tensor());
			}
			
			e.getValue().copyInto(gradIns.get(index));
		});
	}
	
	private void loadSequenceMemories(int index){
		memories.entrySet().forEach(e ->{
			List<Tensor> mems = sequenceMemories.get(e.getKey());
			Tensor state = mems.get(index);
			e.getValue().setMemory(state);
		});	
	}
	
	private Tensor[] getSequenceInputs(int index, List<Tensor>[] inputs){
		Tensor[] ins = new Tensor[inputs.length];
		for(int i=0;i<inputs.length;i++){
			ins[i] = inputs[i].get(index);
		}
		return ins;
	}
	
	
	// let all tags added by NN wrapper precede by "_"
	private String getTag(){
		synchronized(this){
			return "_"+count++;
		}
	}
	
	private String[] addTag(String[] tags, String tag){
		if(tags==null){
			String[] t = new String[1];
			t[0] = tag;
			return t;
		}
		if(tag==null)
			return tags;
		
		int l = tags.length;
		String[] t = new String[l+1];
		System.arraycopy(tags, 0, t, 1, tags.length);
		t[0] = tag;
		return t;
	}
	
	// remove hidden tags (starting with _")
	private String[] removeTags(String[] tags){
		int l = tags.length;
		if(l <= 1){
			return new String[0];
		}
		
		int size = 0;
		for(String s : tags){
			if(!s.startsWith("_")){
				size++;
			}
		}
		if(size == 0){
			return new String[0];
		}
		
		String[] t = new String[size];
		int i = 0;
		for(String s : tags){
			if(!s.startsWith("_")){
				t[i++] = s;
			}
		}
		return t;
	}
	
	void addInterest(UUID moduleId, String tag){
		List<UUID> m = interestedModules.get(tag);
		if(m == null){
			m = new ArrayList<>(1);
			interestedModules.put(tag, m);
		}
		m.add(moduleId);
	}
	
	boolean isInterested(UUID moduleId, String tag){
		List<UUID> m = interestedModules.get(tag);
		if(m == null){
			return false;
		}
		for(UUID id : m){
			if(id.equals(moduleId))
				return true;
		}
		return false;
	}
	
	void removeInterest(UUID moduleId, String tag){
		List<UUID> m = interestedModules.get(tag);
		if(m == null){
			return;
		}
		m.remove(moduleId);
		if(m.size()==0){
			interestedModules.remove(tag);
		}
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
				
				for(String tag : tags){
					if(!tag.startsWith("_")){
						continue;
					}
					
					if(!isInterested(moduleId, tag)){
						continue;
					}
					
					removeInterest(moduleId, tag);
					Progress p = inProgress.get(tag);
					if(p!=null){
						boolean remove = p.resolve(moduleId, output, removeTags(tags));
						if(remove){
							inProgress.remove(tag);
						}
					}
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				if(tags==null || tags.length==0)
					return;
				
				for(String tag : tags){
					if(!tag.startsWith("_")){
						continue;
					}
					
					if(!isInterested(moduleId, tag)){
						continue;
					}
					
					removeInterest(moduleId, tag);
					Progress p = inProgress.remove(tag);
					if(p!=null){
						p.fail(e);
					}
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
				
				for(String tag : tags){
					if(!tag.startsWith("_")){
						continue;
					}
				
					if(!isInterested(moduleId, tag)){
						continue;
					}
					
					removeInterest(moduleId, tag);
					Progress p = inProgress.get(tag);
					if(p!=null){
						boolean remove = p.resolve(moduleId, gradInput, removeTags(tags));
						if(remove){
							inProgress.remove(tag);
						}
					}
				}
			}

			@Override
			public void onError(UUID moduleId, ModuleException e, String... tags) {
				if(tags==null || tags.length==0) {
					return;
				}
				
				for(String tag : tags){
					if(!tag.startsWith("_")){
						continue;
					}
				
					if(!isInterested(moduleId, tag)){
						continue;
					}
					
					removeInterest(moduleId, tag);
					Progress p = inProgress.remove(tag);
					if(p!=null){
						p.fail(e);
					}
				}
			}
		}, propertiesBw);	
		
		Dictionary<String, Object> properties = new Hashtable<String, Object>();
		properties.put("nn.id", nn.id.toString());
		properties.put("aiolos.export", false);
		properties.put("aiolos.unique", true);
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
	public void setOutputLabels(String[] labels) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		UUID outputId = outputs.keySet().iterator().next();
		outputs.get(outputId).setOutputLabels(labels);
		// also store in repository
		storeOutputLabels(outputId, labels);
	}

	@Override
	public void setOutputLabels(UUID outputId, String[] labels) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Output output = outputs.get(outputId);
		if(output==null){
			throw new RuntimeException("This neural network does not have output "+outputId);
		}
		output.setOutputLabels(labels);
		// also store in repository
		storeOutputLabels(outputId, labels);
	}
	
	private void storeOutputLabels(UUID outputId, String[] labels){
		NeuralNetworkDTO dto = repository.loadNeuralNetwork(nn.name);
		ModuleDTO outputDTO = dto.modules.get(outputId);
		String labelString = "[";
		for(int i=0;i<labels.length;i++){
			labelString+=labels[i];
			if(i!=labels.length-1){
				labelString+=", ";
			} else {
				labelString+="]";
			}
		}
		outputDTO.properties.put("labels", labelString);
		repository.storeNeuralNetwork(dto);
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
	public Map<UUID, Memory> getMemories() {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return memories;
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
		
		trainables.values().stream().forEach(t -> t.randomizeParameters());
	}
	
	@Override
	public void randomizeParameters(UUID moduleId){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Trainable t = trainables.get(moduleId);
		if(t == null){
			throw new RuntimeException("This neural network has no trainable module with id "+moduleId);
		}
		t.randomizeParameters();
	}

	@Override
	public void storeParameters(String... tag) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		 
		repository.storeParameters(nn.id, getParameters(), tag);
	}
	
	@Override
	public void storeParameters(Map<UUID, Tensor> params, String... tag) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		if(tag == null)
			repository.storeParameters(nn.id, params); 
		else 
			repository.storeParameters(nn.id, params, tag);
	}

	@Override
	public void storeDeltaParameters(Map<UUID, Tensor> previous, String... tag) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Map<UUID, Tensor> deltaParameters = trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> TensorOps.sub(null,
						e.getValue().getParameters(), previous.get(e.getKey()))));
		if(tag == null)
			repository.accParameters(nn.id, deltaParameters);
		else 
			repository.accParameters(nn.id, deltaParameters, tag);
			
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String... tag) throws Exception {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}

		Map<UUID, Tensor> parameters = tag==null? repository.loadParameters(nn.name) : repository.loadParameters(nn.name, tag);
		setParameters(parameters);
		return parameters;
	}

	@Override
	public void zeroDeltaParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		trainables.values().stream().forEach(Trainable::zeroDeltaParameters);
	}

	@Override
	public void updateParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		trainables.values().stream().forEach(Trainable::updateParameters);
	}
	
	@Override
	public void accGradParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		trainables.values().stream().forEach(Trainable::accGradParameters);
	}
	
	private class Progress {
		
		private final Deferred<NeuralNetworkResult> deferred = new Deferred<>();
		private final UUID[] resultIds;
		private final Map<UUID, Tensor> results = new HashMap<>();

		public Progress(UUID resultId){
			if(resultId == null){
				resultIds = null;
			} else {
				this.resultIds = new UUID[]{resultId};
			}
		}
		
		public Progress(UUID[] resultIds){
			this.resultIds = resultIds;
		}
		
		public synchronized boolean resolve(UUID moduleId, Tensor tensor, String... tags){
			boolean requested = false;
			if(resultIds == null){
				requested = true;
			} else {
				for(UUID id : resultIds){
					if(moduleId.equals(id)){
						requested = true;
					}
				}
			}
			
			if(requested) {
				results.put(moduleId, tensor);
			}
			
			if(resultIds==null || results.size() == resultIds.length){
				NeuralNetworkResult r = new NeuralNetworkResult(results, tags);
				deferred.resolve(r);
				return true;
			}
			
			return false;
		}
		
		public synchronized void fail(Throwable t){
			deferred.fail(t);
		}
		
		public Promise<NeuralNetworkResult> getPromise(){
			return deferred.getPromise();
		}
	}
	
	
	public void resetMemory(int batchSize){
		memories.values().forEach(m -> m.reset(batchSize));
	}
}
