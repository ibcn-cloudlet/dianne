package be.iminds.iot.dianne.nn.module.fork;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class Fork extends AbstractModule {

	protected Map<UUID, Tensor> outputs = new HashMap<UUID, Tensor>();
	protected Map<UUID, Tensor> gradOutputs = new HashMap<UUID, Tensor>();
	
	// this will make sure that one will wait until all prev have given input before forwarding
	// during training
	protected Map<UUID, AtomicBoolean> nextLock = new HashMap<UUID, AtomicBoolean>();
	
	protected Map<Module, AtomicBoolean> nextsBusy = new HashMap<Module, AtomicBoolean>(); 
	
	protected UUID[] nextIds;
	
	public Fork(TensorFactory factory) {
		super(factory);
	}
	
	public Fork(TensorFactory factory, UUID id) {
		super(factory, id);
	}

	@Override
	protected void callNext(){
		// call all next
		for(int i=0; i< next.length;i++){
			UUID id = nextIds[i];
			Module m = next[i];
			nextsBusy.get(m).set(true);
			
			executor.execute(new ForwardForkRunnable(m, outputs.get(id), tags));
		}
	}
	
	@Override
	public synchronized void forward(final UUID moduleId, final Tensor input, final String... tags) {
		// skip or block when nexts are not ready processing previous output of this module
		synchronized(nextsBusy){
			while(nextBusy()){
				// next is busy, either block or skip
				if(mode.contains(Mode.SKIP)){
					System.out.println("Module "+id+" skipped input");
					return;
				} else {
					// default mode BLOCKING
					try {
						nextsBusy.wait();
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
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput, final String... tags) {
		this.tags = tags;
		this.gradOutputs.put(moduleId, gradOutput);
		
		// when wait-for-all mode, wait until all gradOutput is updated
		if(mode.contains(Mode.WAIT_FOR_ALL) && next!=null && next.length>1){
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
		
		backward();
		
		// backward on separate thread
		if(prev!=null)
			callPrevious();
		
		if(bwListeners.size()>0)
			notifyBackwardListeners();
	}
	
	@Override
	public void setNext(final Module... next) {
		if(next==null){
			this.next = null;
			this.nextIds = null;
		} else {
			this.next = next;
			this.nextIds = new UUID[next.length];
			this.nextLock.clear();
			this.nextsBusy.clear();
			for(int i=0;i<next.length;i++){
				UUID id = next[i].getId();
				// make sure that UUIDs are in keys
				// TODO better fix for this?
				this.nextIds[i] = id;
				this.outputs.put(id, null);
				this.gradOutputs.put(id, null);
				this.nextLock.put(id, new AtomicBoolean(false));
				this.nextsBusy.put(next[i], new AtomicBoolean(false));
			}
		}
	}
	
	protected final class ForwardForkRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		
		public ForwardForkRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
		}
		
		public void run(){
			m.forward(id, tensor, tags);
			synchronized(nextsBusy){
				nextsBusy.get(m).set(false);
				nextsBusy.notifyAll();
			}
		}
	}
	
	protected boolean nextBusy(){
		int i = 0;
		for(AtomicBoolean a : nextsBusy.values()){
			if(a.get()){
				return true;
			}
			i++;
		}
		return false;
	}
}
