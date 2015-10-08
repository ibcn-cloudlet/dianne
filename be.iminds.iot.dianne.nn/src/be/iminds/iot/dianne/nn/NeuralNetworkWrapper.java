package be.iminds.iot.dianne.nn;

import java.util.Arrays;
import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
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
	private final static String SYNC_TAG = "sync"; 
	
	private final DianneRepository repository;
	private final TensorFactory factory;
	
	private final NeuralNetworkInstanceDTO nn;
	private Map<UUID, Module> modules;
	private Map<UUID, Input> inputs;
	private Map<UUID, Output> outputs;
	private Map<UUID, Preprocessor> preprocessors;
	private Map<UUID, Trainable> trainables;
	
	private final BundleContext context;
	private WaitingForwardListener forwardListener;
	private WaitingBackwardListener backwardListener;
	private Map<ForwardListener, String> delegateFw = new HashMap<ForwardListener, String>();
	private Map<BackwardListener, String> delegateBw = new HashMap<BackwardListener, String>();
	private ServiceRegistration<ForwardListener> forwardListenerReg;
	private ServiceRegistration<BackwardListener> backwardListenerReg;
	private ServiceRegistration<NeuralNetwork> nnReg;
	
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
	public Tensor forward(Tensor in, String... tags) {
		if(inputs.size() > 1){
			throw new RuntimeException("This neural network has more than one input");
		}
		return forward(inputs.keySet().iterator().next(), outputs.keySet().iterator().next(), in, tags);
	}
	
	@Override
	public synchronized Tensor forward(UUID inputId, UUID outputId, Tensor in,
			String... tags) {
		Tensor output = null;
		synchronized(forwardListener){
			if(!outputs.containsKey(outputId)){
				throw new RuntimeException("This neural network does not have output "+inputId);
			}
			
			Input input = inputs.get(inputId);
			if(input==null){
				throw new RuntimeException("This neural network does not have input "+inputId);
			}
			
			input.input(in, addTag(tags, SYNC_TAG));
			try {
				forwardListener.wait(outputId);
			} catch (InterruptedException e) {
			}
			output = forwardListener.get();;
		}
		return output;
	}
	
	@Override
	public void aforward(Tensor in, String... tags){
		aforward((ForwardListener)null, in, tags);
	}	

	@Override
	public void aforward(ForwardListener callback, Tensor in, String... tags){
		if(inputs.size() > 1){
			throw new RuntimeException("This neural network has more than one input");
		}
		String tag = getTag(callback);
		inputs.values().iterator().next().input(in, addTag(tags, tag));
	}
	
	@Override
	public void aforward(UUID inputId, Tensor in, String... tags) {
		aforward((ForwardListener)null, inputId, in, tags);
	}
	
	@Override
	public void aforward(ForwardListener callback, UUID inputId, Tensor in,
			String... tags) {
		String tag = getTag(callback);
		Input input = inputs.get(inputId);
		if(input==null){
			throw new RuntimeException("This neural network does not have input "+inputId);
		}
		input.input(in, addTag(tags, tag));
	}

	@Override
	public Tensor backward(Tensor gradOut, String... tags) {
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		return backward(outputs.keySet().iterator().next(),
				inputs.keySet().iterator().next(), gradOut, tags);
	}
	
	@Override
	public synchronized Tensor backward(UUID outputId, UUID inputId, Tensor gradOut,
			String... tags) {
		Tensor gradInput = null;
		synchronized(backwardListener){
			if(!inputs.containsKey(inputId)){
				throw new RuntimeException("This neural network does not have input "+inputId);
			}
			
			Output output = outputs.get(outputId);
			if(output==null){
				throw new RuntimeException("This neural network does not have output "+outputId);
			}
			
			output.backpropagate(gradOut, addTag(tags, SYNC_TAG));
			try {
				backwardListener.wait(inputId);
			} catch (InterruptedException e) {
			}
			gradInput = backwardListener.get();
		}
		return gradInput;
	}
	
	@Override
	public void abackward(Tensor gradOut, String... tags){
		abackward((BackwardListener)null, gradOut, tags);
	}
	
	@Override
	public void abackward(BackwardListener callback, Tensor gradOut, String... tags){
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		String tag = getTag(callback);
		outputs.values().iterator().next().backpropagate(gradOut, addTag(tags, tag));
	}
	
	@Override
	public void abackward(UUID outputId,
			Tensor gradOut, String... tags) {
		abackward((BackwardListener)null, outputId,gradOut, tags);
	}
	
	@Override
	public void abackward(BackwardListener callback, UUID outputId,
			Tensor gradOut, String... tags) {
		String tag = getTag(callback);
		Output output = outputs.get(outputId);
		if(output==null){
			throw new RuntimeException("This neural network does not have output "+outputId);
		}
		output.backpropagate(gradOut, addTag(tags, tag));
	}
	
	private String getTag(ForwardListener l){
		String tag = null;
		if(l!=null){
			if(!delegateFw.containsKey(l)){
				tag = ""+(count++);
				delegateFw.put(l, tag);
			} else {
				tag = delegateFw.get(l);
			}
		}
		return tag;
	}
	
	private String getTag(BackwardListener l){
		String tag = null;
		if(l!=null){
			if(!delegateBw.containsKey(l)){
				tag = ""+(count++);
				delegateBw.put(l, tag);
			} else {
				tag = delegateBw.get(l);
			}
		}
		return tag;
	}
	
	private String[] addTag(String[] tags, String tag){
		if(tag==null)
			return tags;
		
		int l = tags.length;
		String[] t = Arrays.copyOf(tags, l+1);
		t[l] = tag;
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
	
	private boolean containsTag(String[] tags, String tag){
		for(String t : tags){
			if(t.equals(tag)){
				return true;
			}
		}
		return false;
	}
	
	void register(){
		Dictionary<String, Object> propertiesFw = new Hashtable<String, Object>();
		propertiesFw.put("targets", new String[]{nn.id.toString()});
		propertiesFw.put("aiolos.unique", true);
		forwardListener = new WaitingForwardListener();
		forwardListenerReg = context.registerService(ForwardListener.class, forwardListener, propertiesFw);
	
		Dictionary<String, Object> propertiesBw = new Hashtable<String, Object>();
		propertiesBw.put("targets", new String[]{nn.id.toString()});
		propertiesBw.put("aiolos.unique", true);
		backwardListener = new WaitingBackwardListener();
		backwardListenerReg = context.registerService(BackwardListener.class, backwardListener, propertiesBw);	
		
		Dictionary<String, Object> properties = new Hashtable<String, Object>();
		properties.put("nn.id", nn.id.toString());
		properties.put("aiolos.export", false);
		nnReg = context.registerService(NeuralNetwork.class, this, properties);	
	}
	
	void unregister(){
		forwardListenerReg.unregister();
		backwardListenerReg.unregister();
		nnReg.unregister();
	}

	@Override
	public Input getInput(){
		if(inputs.size() > 1){
			throw new RuntimeException("This neural network has more than one input");
		}
		return inputs.values().iterator().next();
	}
	
	@Override
	public Map<UUID, Input> getInputs() {
		return inputs;
	}

	@Override
	public Output getOutput(){
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		return outputs.values().iterator().next();
	}
	
	@Override
	public Map<UUID, Output> getOutputs() {
		return outputs;
	}

	@Override
	public String[] getOutputLabels() {
		if(outputs.size() > 1){
			throw new RuntimeException("This neural network has more than one output");
		}
		return outputs.values().iterator().next().getOutputLabels();
	}

	@Override
	public String[] getOutputLabels(UUID outputId) {
		Output output = outputs.get(outputId);
		if(output==null){
			throw new RuntimeException("This neural network does not have output "+outputId);
		}
		return output.getOutputLabels();
	}

	@Override
	public Map<UUID, Trainable> getTrainables() {
		return trainables;
	}

	@Override
	public Map<UUID, Preprocessor> getPreprocessors() {
		return preprocessors;
	}
	
	@Override
	public Map<UUID, Module> getModules() {
		return modules;
	}

	private class WaitingForwardListener implements ForwardListener {

		private Tensor out;
		private UUID waitId;
		
		public Tensor get(){
			return out;
		}
		
		public void wait(UUID moduleId) throws InterruptedException {
			this.waitId = moduleId;
			this.wait();
		}
		
		@Override
		public void onForward(UUID moduleId, Tensor output, String... tags) {
			delegateFw.entrySet().stream()
				.filter(e -> containsTag(tags, e.getValue()))
				.forEach(e -> e.getKey().onForward(moduleId, output, removeTag(tags, e.getValue())));
			
			if(containsTag(tags, SYNC_TAG)){
				out = output;
				synchronized(forwardListener){
					if(moduleId.equals(waitId))
						forwardListener.notifyAll();
				}
			}
		}
	}
	
	private class WaitingBackwardListener implements BackwardListener {
		
		private Tensor gradInput;
		private UUID waitId;
		
		public Tensor get(){
			return gradInput;
		}
		
		public void wait(UUID moduleId) throws InterruptedException {
			this.waitId = moduleId;
			this.wait();
		}
		
		@Override
		public void onBackward(UUID moduleId, Tensor gradIn, String... tags) {
			delegateBw.entrySet().stream()
				.filter(e -> containsTag(tags, e.getValue()))
				.forEach(e -> e.getKey().onBackward(moduleId, gradIn, removeTag(tags, e.getValue())));
			
			if(containsTag(tags, SYNC_TAG)){
				gradInput = gradIn;
				synchronized(backwardListener){
					if(moduleId.equals(waitId))
						backwardListener.notifyAll();
				}
			}
		}
	}

	@Override
	public void setParameters(Map<UUID, Tensor> parameters) {
		parameters.entrySet().forEach(e -> {
			Trainable t = trainables.get(e.getKey());
			if(t!=null)
				t.setParameters(e.getValue());
		});
	}

	@Override
	public void setParameters(UUID moduleId, Tensor parameters) {
		Trainable t = trainables.get(moduleId);
		if(t!=null)
			t.setParameters(parameters);
	}

	@Override
	public Map<UUID, Tensor> getParameters(){
		return trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().getParameters()));
	}
	
	@Override
	public void resetParameters(){
		trainables.values().stream().forEach(t -> t.reset());
	}

	@Override
	public void storeParameters(String... tag) {
		repository.storeParameters(nn.id, getParameters(), tag);
	}

	@Override
	public void storeDeltaParameters(Map<UUID, Tensor> previous, String... tag) {
		Map<UUID, Tensor> deltaParameters = trainables.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> factory.getTensorMath().sub(null,
						e.getValue().getParameters(), previous.get(e.getKey()))));

		repository.accParameters(nn.id, deltaParameters, tag);
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String... tag) throws Exception {
		Map<UUID, Tensor> parameters = repository.loadParameters(nn.name, tag);
		setParameters(parameters);
		return parameters;
	}
}
