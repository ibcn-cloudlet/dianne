package be.iminds.iot.dianne.nn.module.fork;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class Fork extends AbstractModule {

	protected Map<UUID, Tensor> outputs = new HashMap<UUID, Tensor>();
	protected Map<UUID, Tensor> gradOutputs = new HashMap<UUID, Tensor>();
	
	// this will make sure that one will wait until all prev have given input before forwarding
	protected boolean sync = true;
	protected Map<UUID, AtomicBoolean> nextLock = new HashMap<UUID, AtomicBoolean>();
	
	public Fork(TensorFactory factory) {
		super(factory);
	}
	
	public Fork(TensorFactory factory, UUID id) {
		super(factory, id);
	}

	protected void callNext(){
		// call all next, ForwardForkRunnable will make sure each gets part of the outputs
		for(Runnable r : next){
			executor.execute(r);
		}
	}
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput) {
		this.gradOutputs.put(moduleId, gradOutput);
		
		// when synchronized, wait until all gradOutput is updated
		if(sync && next!=null && next.length>1){
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
		} else {
			this.next = new ForwardForkRunnable[next.length];
			for(int i=0;i<next.length;i++){
				UUID id = next[i].getId();
				// make sure that UUIDs are in keys
				// TODO better fix for this?
				this.outputs.put(id, null);
				this.gradOutputs.put(id, null);
				this.nextLock.put(id, new AtomicBoolean(false));
				this.next[i] = new ForwardForkRunnable(next[i], id);
			}
		}
	}
	
	private final class ForwardForkRunnable implements Runnable {
		private final Module m;
		private final UUID nextId;
		
		public ForwardForkRunnable(Module m, UUID id){
			this.m = m;
			this.nextId = id;
		}
		
		public void run(){
			m.forward(id, outputs.get(nextId));
		}
	}
	
}
