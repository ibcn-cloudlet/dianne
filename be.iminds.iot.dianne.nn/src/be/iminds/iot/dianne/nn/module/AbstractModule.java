package be.iminds.iot.dianne.nn.module;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
	//protected Module[] next;
	protected Map<UUID, Module> next;
	// The prev module references
	//protected Module[] prev;
	protected Map<UUID, Module> prev;
	
	// this will make sure that one will wait until all prev have given input before forwarding
	protected boolean sync = true;
	protected Map<UUID, AtomicBoolean> nextLock;
	protected Map<UUID, AtomicBoolean> prevLock;
	
	// Thread executor to perform calculations on
	private ExecutorService executor = Executors.newSingleThreadExecutor();
	
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
	
	protected Runnable forward = new Runnable(){
		public void run(){
			if(next!=null){
				for(Module m : next.values())
					m.forward(AbstractModule.this.id, AbstractModule.this.output);
			}
		}
	};
	
	protected Runnable backward = new Runnable(){
		public void run(){
			if(prev!=null){
				for(Module m : prev.values()){
					m.backward(AbstractModule.this.id, AbstractModule.this.gradInput);
				}
			}
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
		forward(moduleId);
		
		// forward on separate thread
		if(sync && prev!=null && prev.size()>1){
			synchronized(nextLock){
				nextLock.get(moduleId).set(true);
				for(AtomicBoolean b : nextLock.values()){
					if(!b.get()){
						return;
					}
				}
				for(AtomicBoolean b : nextLock.values()){
					b.set(false);
				}
			}
		} 
		
		executor.execute(forward);
	
		notifyForwardListeners();
	}
	
	protected abstract void forward(UUID from);
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput) {
		this.gradOutput = gradOutput;
		
		// calculates new gradInputs
		//System.out.println("Backward "+AbstractModule.this.getClass().getName()+" "+id);
		backward(moduleId);
		
		if(sync && next!=null && next.size()>1){
			synchronized(prevLock){
				prevLock.get(moduleId).set(true);
				for(AtomicBoolean b : prevLock.values()){
					if(!b.get()){
						return;
					}
				}
				for(AtomicBoolean b : prevLock.values()){
					b.set(false);
				}
			}
		} 
		
		// backward on separate thread
		executor.execute(backward);
		
		notifyBackwardListeners();
	}
	
	protected abstract void backward(UUID from);

	@Override
	public void setNext(final Module... next) {
		if(next==null){
//			System.out.println("Reset next of "+id);
			this.next = null;
		} else {
//			System.out.println("Set "+id+"->\t"+next[0].getId());
			this.next = new HashMap<UUID, Module>();
			this.prevLock = new HashMap<UUID, AtomicBoolean>();
			for(Module m : next){
				UUID nextId = m.getId();
				this.next.put(nextId, m);
				this.prevLock.put(nextId, new AtomicBoolean(false));
			}
		}
	}

	@Override
	public void setPrevious(final Module... prev) {
		if(prev==null){
//			System.out.println("Reset prev of "+id);
			this.prev = null;
		} else {
//			System.out.println("Set "+id+"\t<-"+prev[0].getId());
			this.prev = new HashMap<UUID, Module>();
			this.nextLock = new HashMap<UUID, AtomicBoolean>();
			for(Module m : prev){
				UUID prevId = m.getId();
				this.prev.put(prevId, m);
				this.nextLock.put(prevId, new AtomicBoolean(false));
			}
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
				if(fwdListeners.size()>0){
					synchronized (fwdListeners) {
						for(ForwardListener f : fwdListeners){
							f.onForward(AbstractModule.this.output);
						}
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
				if(bwListeners.size()>0){
					synchronized (bwListeners) {
						for(BackwardListener b : bwListeners){
							b.onBackward(AbstractModule.this.output);
						}
					}
				}
			}
		};
		executor.execute(r);
	}
}
