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
		
		parameters.entrySet().forEach(e -> {
			Trainable t = trainables.get(e.getKey());
			if(t!=null)
				t.setParameters(e.getValue());
		});
	}

	@Override
	public void setParameters(UUID moduleId, Tensor parameters) {
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		Trainable t = trainables.get(moduleId);
		if(t!=null)
			t.setParameters(parameters);
	}

	@Override
	public Map<UUID, Tensor> getParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		return trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().getParameters()));
	}
	
	@Override
	public void resetParameters(){
		if(!valid){
			throw new RuntimeException("This neural network object is no longer valid");
		}
		
		trainables.values().stream().forEach(t -> t.reset());
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
