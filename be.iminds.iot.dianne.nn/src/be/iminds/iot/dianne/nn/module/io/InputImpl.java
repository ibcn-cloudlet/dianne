package be.iminds.iot.dianne.nn.module.io;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class InputImpl extends AbstractModule implements Input {

	public InputImpl(TensorFactory factory) {
		super(factory);
	}

	public InputImpl(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	public void input(Tensor input){
		forward(this.id, input);
	}
	
	@Override
	protected void forward(UUID from) {
		output = input;
	}

	@Override
	protected void backward(UUID from) {
		gradInput = gradOutput;
	}

	@Override
	public void setPrevious(final Module... prev) {
		System.out.println("Input cannot have previous modules");
	}
}
