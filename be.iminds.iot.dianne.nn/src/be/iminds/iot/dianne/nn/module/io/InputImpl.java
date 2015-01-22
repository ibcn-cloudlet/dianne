package be.iminds.iot.dianne.nn.module.io;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.InputListener;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class InputImpl extends AbstractModule implements Input {

	private List<InputListener> listeners = Collections.synchronizedList(new ArrayList<InputListener>());

	public InputImpl(TensorFactory factory) {
		super(factory);
	}
	
	@Override
	public void input(Tensor input){
		forward(this.id, input);
	}
	
	@Override
	protected void forward() {
		output = input;
	}

	@Override
	protected void backward() {
		gradInput = gradOutput;
		
		notifyListeners();
	}

	@Override
	public void setPrevious(final Module... prev) {
		System.out.println("Input cannot have previous modules");
	}

	@Override
	public void addInputListener(InputListener listener) {
		listeners.add(listener);
	}

	@Override
	public void removeInputListener(InputListener listener) {
		listeners.remove(listener);
	}

	private void notifyListeners(){
		synchronized(listeners){
			for(InputListener l : listeners){
				l.onBackward(output);
			}
		}
	}
	
}
