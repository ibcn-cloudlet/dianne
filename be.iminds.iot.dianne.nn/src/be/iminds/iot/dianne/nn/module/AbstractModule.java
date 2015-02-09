package be.iminds.iot.dianne.nn.module;

import java.util.ArrayList;
import java.util.Collections;
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
	protected Module[] next;
	// The prev module references
	protected Module[] prev;
	
	// Thread executor to perform calculations on
	protected ExecutorService executor = Executors.newSingleThreadExecutor();
	
	// Listeners
	protected List<ForwardListener> fwdListeners = Collections.synchronizedList(new ArrayList<ForwardListener>());
	protected List<BackwardListener> bwListeners = Collections.synchronizedList(new ArrayList<BackwardListener>());

	
	public void setExecutorService(ExecutorService executor){
		List<Runnable> todo = this.executor.shutdownNow();
		this.executor = executor;
		for(Runnable r : todo){
			this.executor.execute(r);
		}
	}
	
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

	protected final Runnable forward = new Runnable(){
		public void run(){
			if(next!=null){
				callNext();
			}
		}
	};
	
	protected void callNext(){
		// default AbstractModule just assumes one next and one previous, use Fork otherwise
		next[0].forward(this.id, this.output);
	}
	
	protected final Runnable backward = new Runnable(){
		public void run(){
			if(prev!=null){
				callPrevious();
			}
		}
	};
	
	protected void callPrevious(){
		// default AbstractModule just assumes one next and one previous, use Join otherwise
		prev[0].backward(this.id, this.gradInput);
	}
	
	@Override
	public void forward(final UUID moduleId, final Tensor input) {
		this.input = input;
		
		// calculates new outputs
		forward();
		
		executor.execute(forward);
	
		if(fwdListeners.size()>0)
			notifyForwardListeners();
	}
	
	protected abstract void forward();
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput) {
		this.gradOutput = gradOutput;
		
		// calculates new gradInputs
		backward();
		
		// backward on separate thread
		executor.execute(backward);
		
		if(bwListeners.size()>0)
			notifyBackwardListeners();
	}
	
	protected abstract void backward();

	@Override
	public void setNext(final Module... next) {
		if(next==null){
			this.next = null;
		} else {
			this.next = next;
		}
	}

	@Override
	public void setPrevious(final Module... prev) {
		if(prev==null){
			this.prev = null;
		} else {
			this.prev = prev;
		}
	}

	public void addForwardListener(ForwardListener listener){
		fwdListeners.add(listener);
	}
	
	public void removeForwardListener(ForwardListener listener){
		fwdListeners.remove(listener);
	}
	
	protected void notifyForwardListeners(){
		Runnable r = new Runnable() {
			@Override
			public void run() {
				synchronized (fwdListeners) {
					for(ForwardListener f : fwdListeners){
						f.onForward(AbstractModule.this.output);
					}
				}
			}
		};
		executor.execute(r);
	}
	
	public void addBackwardListener(BackwardListener listener){
		bwListeners.add(listener);
	}
	
	public void removeBackwardListener(BackwardListener listener){
		bwListeners.remove(listener);
	}
	
	protected void notifyBackwardListeners(){
		Runnable r = new Runnable() {
			@Override
			public void run() {
				synchronized (bwListeners) {
					for(BackwardListener b : bwListeners){
						b.onBackward(AbstractModule.this.gradInput);
					}
				}
			}
		};
		executor.execute(r);
	}
}
