package be.iminds.iot.dianne.nn.module;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import be.iminds.iot.dianne.nn.module.Module.Mode;
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
	protected Runnable[] next;
	// The prev module references
	protected Runnable[] prev;
	
	// Thread executor to perform calculations on
	protected ExecutorService executor = Executors.newSingleThreadExecutor();
	
	// Listeners
	protected List<ForwardListener> fwdListeners = Collections.synchronizedList(new ArrayList<ForwardListener>());
	protected List<BackwardListener> bwListeners = Collections.synchronizedList(new ArrayList<BackwardListener>());

	// Mode
	protected Mode mode;
	
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
	
	protected void callNext(){
		// default AbstractModule just assumes one next and one previous, use Fork otherwise
		executor.execute(next[0]);
	}
	
	protected void callPrevious(){
		// default AbstractModule just assumes one next and one previous, use Join otherwise
		executor.execute(prev[0]);
	}
	
	@Override
	public synchronized void forward(final UUID moduleId, final Tensor input) {
		this.input = input;
		
		// calculates new outputs
		if(output!=null){
			synchronized(output){
				synchronized(input){
					forward();
				}
			}
		} else {
			synchronized(input){
				forward();
			}
		}
		
		if(next!=null)
			callNext();
	
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
		if(prev!=null)
			callPrevious();
		
		if(bwListeners.size()>0)
			notifyBackwardListeners();
	}
	
	protected abstract void backward();

	@Override
	public void setNext(final Module... next) {
		if(next==null){
			this.next = null;
		} else {
			this.next = new ForwardRunnable[next.length];
			for(int i=0;i<next.length;i++){
				this.next[i] = new ForwardRunnable(next[i]);
			}
		}
	}

	@Override
	public void setPrevious(final Module... prev) {
		if(prev==null){
			this.prev = null;
		} else {
			this.prev = new BackwardRunnable[prev.length];
			for(int i=0;i<prev.length;i++){
				this.prev[i] = new BackwardRunnable(prev[i]);
			}
		}
	}

	@Override
	public void setMode(Mode mode){
		this.mode = mode;
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
	
	private final class ForwardRunnable implements Runnable {
		private final Module m;
		
		public ForwardRunnable(Module m){
			this.m = m;
		}
		
		public void run(){
			m.forward(id, output);
		}
	}
	
	private final class BackwardRunnable implements Runnable {
		private final Module m;
		
		public BackwardRunnable(Module m){
			this.m = m;
		}
		
		public void run(){
			m.backward(id, gradInput);
		}
	}
}
