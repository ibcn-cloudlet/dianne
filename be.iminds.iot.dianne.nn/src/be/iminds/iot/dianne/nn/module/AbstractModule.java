package be.iminds.iot.dianne.nn.module;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractModule implements Module {

	// the factory for this module
	protected final TensorFactory factory;
	
	// the UUID of this module
	protected final UUID id;
	
	// the latest input given by the previous module
	// contains the Tensor (reference) given by previous
	protected Tensor input;
	// Tensor to put the results in of backward calculation:
	// calculates the input gradients based on the latest output gradients
	protected Tensor gradInput;

	// Tensor to put the results from from forward calculation
	// calculates the feed forward of the module
	protected Tensor output;
	// the latest gradOutputs given by previous modules
	// contains the Tensor (reference) given by previous
	protected Tensor gradOutput;
	
	// The next module reference
	protected Module next;
	// The prev module references
	protected Module prev;
	
	// Thread executor to perform calculations on
	private ExecutorService executor = Executors.newSingleThreadExecutor();
	
	public void setExecutorService(ExecutorService executor){
		List<Runnable> todo = this.executor.shutdownNow();
		this.executor = executor;
		for(Runnable r : todo){
			this.executor.execute(r);
		}
	}
	
	private Runnable forward = new Runnable(){
		public void run(){
			if(next!=null)
				next.forward(AbstractModule.this.id, AbstractModule.this.output);
		}
	};
	
	private Runnable backward = new Runnable(){
		public void run(){
			if(prev!=null)
				prev.backward(AbstractModule.this.id, AbstractModule.this.gradInput);
		}
	};

	public AbstractModule(TensorFactory factory) {
		this.id = UUID.randomUUID();
		this.factory = factory;
	}
	
	public AbstractModule(TensorFactory factory, UUID id) {
		this.id = id;
		this.factory = factory;
	}
	
	@Override
	public UUID getId() {
		return id;
	}

	@Override
	public void forward(final UUID moduleId, final Tensor input) {
		this.input = input;
		
		// calculates new outputs
		//System.out.println("Forward "+AbstractModule.this.getClass().getName()+" "+id);
		forward();
		
		// forward on separate thread
		executor.execute(forward);
	}
	
	protected abstract void forward();
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput) {
		this.gradOutput = gradOutput;
		
		// calculates new gradInputs
		//System.out.println("Backward "+AbstractModule.this.getClass().getName()+" "+id);
		backward();
		
		// backward on separate thread
		executor.execute(backward);
	}
	
	protected abstract void backward();

	@Override
	public void setNext(final Module... next) {
		if(next==null){
//			System.out.println("Reset next of "+id);
			this.next = null;
		} else {
//			System.out.println("Set "+id+"->\t"+next[0].getId());
			this.next = next[0];
		}
	}

	@Override
	public void setPrevious(final Module... prev) {
		if(prev==null){
//			System.out.println("Reset prev of "+id);
			this.prev = null;
		} else {
//			System.out.println("Set "+id+"\t<-"+prev[0].getId());
			this.prev = prev[0];
		}
	}

}
