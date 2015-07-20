package be.iminds.iot.dianne.api.nn.module;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractModule implements Module {

	// the factory for this module
	protected final TensorFactory factory;
	
	// the UUID of this module
	protected final UUID id;
	
	// the tags to forward/backward
	// these are set in each forward/backward call and can be
	// adapter/filtered by the actual module implementations
	protected String[] tags;
	
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
	
	// Boolean that indicates whether the next Module is still busy processing this module output
	// Can be used to either skip or block here
	protected AtomicBoolean nextBusy = new AtomicBoolean();
	
	// Thread executor to perform calculations on
	protected ExecutorService executor = Executors.newSingleThreadExecutor();
	
	// Listeners
	protected Set<ForwardListener> fwdListeners = Collections.synchronizedSet(new HashSet<ForwardListener>());
	protected Set<BackwardListener> bwListeners = Collections.synchronizedSet(new HashSet<BackwardListener>());

	// Mode
	protected EnumSet<Mode> mode = EnumSet.of(Mode.BLOCKING);
	
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
		nextBusy.set(true);
		// default AbstractModule just assumes one next and one previous, use Fork otherwise
		executor.execute(new ForwardRunnable(next[0], output, tags));
	}
	
	protected void callPrevious(){
		// default AbstractModule just assumes one next and one previous, use Join otherwise
		executor.execute(new BackwardRunnable(prev[0], gradInput, tags));
	}
	
	@Override
	public synchronized void forward(final UUID moduleId, final Tensor input, final String... tags) {
		// skip or block when next is not ready processing previous output of this module
		synchronized(nextBusy){
			if(nextBusy.get()){
				// next is busy, either block or skip
				if(mode.contains(Mode.SKIP)){
					System.out.println("Module "+id+" skipped input");
					return;
				} else {
					// default mode BLOCKING
					try {
						nextBusy.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		
		this.input = input;
		this.tags = tags;
		
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
	public void backward(final UUID moduleId, final Tensor gradOutput, final String... tags) {
		this.gradOutput = gradOutput;
		this.tags = tags;
		
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
		this.next = next;
	}

	@Override
	public void setPrevious(final Module... prev) {
		this.prev = prev;
	}

	@Override
	public void setMode(EnumSet<Mode> mode){
		this.mode = mode;
	}
	
	public void addForwardListener(ForwardListener listener){
		fwdListeners.add(listener);
	}
	
	public void removeForwardListener(ForwardListener listener){
		fwdListeners.remove(listener);
	}
	
	protected void notifyForwardListeners(){
		final Tensor outputCopy = output.copyInto(null);
		final String[] tagsCopy = Arrays.copyOf(tags, tags.length);
		Runnable r = new Runnable() {
			@Override
			public void run() {
				synchronized (fwdListeners) {
					for(ForwardListener f : fwdListeners){
						f.onForward(outputCopy, tagsCopy);
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
		final Tensor gradInputCopy = gradInput.copyInto(null);
		final String[] tagsCopy = Arrays.copyOf(tags, tags.length);
		Runnable r = new Runnable() {
			@Override
			public void run() {
				synchronized (bwListeners) {
					for(BackwardListener b : bwListeners){
						b.onBackward(gradInputCopy, tagsCopy);
					}
				}
			}
		};
		executor.execute(r);
	}
	
	protected final class ForwardRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		
		public ForwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
		}
		
		public void run(){
			m.forward(id, tensor, tags);
			
			synchronized(nextBusy){
				nextBusy.set(false);
				nextBusy.notifyAll();
			}
		}
	}
	
	protected final class BackwardRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		
		public BackwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
		}
		
		public void run(){
			m.backward(id, tensor, tags);
		}
	}
	
}
