package be.iminds.iot.dianne.nn.module;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractModule implements Module {

	// the UUID of this module
	protected final UUID id;
	
	// the latest input given by any of the previous modules
	// contains the Tensor (reference) given by previous
	protected Map<UUID, Tensor> inputs = new HashMap<UUID, Tensor>();
	// Tensors to put the results in of backward calculation:
	// calculates the input gradients based on the latest output gradients
	protected Map<UUID, Tensor> gradInputs = new HashMap<UUID, Tensor>();

	// Tensor to put the results from from forward calculation
	// calculates the feed forward of the module
	protected Map<UUID, Tensor> outputs = new HashMap<UUID, Tensor>();
	// the latest gradOutputs given by previous modules
	// contains the Tensor (reference) given by previous
	protected Map<UUID, Tensor> gradOutputs = new HashMap<UUID, Tensor>();
	
	// The next module references
	protected Map<UUID, Module> nexts = new HashMap<UUID, Module>();
	// The prev module references
	protected Map<UUID, Module> prevs = new HashMap<UUID, Module>();
	
	// Thread executor to perform calculations on
	private ExecutorService executor = Executors.newSingleThreadExecutor();
	
	private Runnable forward = new Runnable(){
		public void run(){
			// calculates new outputs
			forward();
			
			// TODO synchronize?
			for(Module m : nexts.values()){
				// TODO check if output available?
				m.forward(AbstractModule.this.id,
						AbstractModule.this.outputs.get(m.getId()));
			}
		}
	};
	
	private Runnable backward = new Runnable(){
		public void run(){
			// calculates new gradInputs
			backward();
			
			// TODO synchronize?
			for(Module m : prevs.values()){
				// TODO check if gradInput available?
				m.backward(AbstractModule.this.id,
						AbstractModule.this.gradInputs.get(m.getId()));
			}
		}
	};
	
	public AbstractModule() {
		this.id = UUID.randomUUID(); // TODO meaningful uuid?
	}
	
	@Override
	public UUID getId() {
		return id;
	}

	@Override
	public void forward(final UUID moduleId, final Tensor input) {
		inputs.put(moduleId, input);
		
		executor.execute(forward);
	}
	
	protected abstract void forward();
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput) {
		gradOutputs.put(moduleId, gradOutput);
		
		executor.execute(backward);
	}
	
	protected abstract void backward();

	@Override
	public void addNext(final Module... next) {
		for(Module n : next){
			this.nexts.put(n.getId(), n);
		}
	}

	@Override
	public void removeNext(final Module... next) {
		for(Module n : next){
			this.nexts.remove(n.getId());
		}
	}

	@Override
	public void addPrevious(final Module... prev) {
		for(Module p : prev){
			this.prevs.put(p.getId(), p);
		}
	}

	@Override
	public void removePrevious(final Module... prev) {
		for(Module p : prev){
			this.prevs.remove(p.getId());
		}
	}

}
