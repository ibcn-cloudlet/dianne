package be.iminds.iot.dianne.nn.module.io;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.OutputListener;
import be.iminds.iot.dianne.tensor.Tensor;

public class OutputImpl extends AbstractModule implements Output {

	private List<OutputListener> listeners = Collections.synchronizedList(new ArrayList<OutputListener>());
	
	@Override
	public Tensor getOutput(){
		return output;
	}
	
	@Override
	public void backpropagate(Tensor gradOutput) {
		backward(this.id, gradOutput);
	}
	
	@Override
	protected void forward() {
		output = input;
		notifyListeners();
	}

	@Override
	protected void backward() {
		gradInput = gradOutput;
	}
	
	@Override
	public void setNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}

	@Override
	public void addOutputListener(OutputListener listener) {
		listeners.add(listener);
	}

	@Override
	public void removeOutputListener(OutputListener listener) {
		listeners.remove(listener);
	}

	private void notifyListeners(){
		synchronized(listeners){
			for(OutputListener l : listeners){
				l.onForward(output);
			}
		}
	}
}
