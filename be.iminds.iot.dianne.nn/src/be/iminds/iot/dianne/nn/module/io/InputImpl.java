package be.iminds.iot.dianne.nn.module.io;

import java.util.EnumSet;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class InputImpl extends AbstractModule implements Input {

	public InputImpl(TensorFactory factory) {
		super(factory);
		
		// Set default mode to SKIP for input
		this.mode = EnumSet.of(Mode.SKIP);
	}

	public InputImpl(TensorFactory factory, UUID id) {
		super(factory, id);
		
		// Set default mode to SKIP for input
		this.mode = EnumSet.of(Mode.SKIP);
	}
	
	@Override
	public void input(final Tensor input, final String... tags){
		forward(this.id, input, tags);
	}
	
	@Override
	protected void forward() {
		output = input;
	}

	@Override
	protected void backward() {
		gradInput = gradOutput;
	}

	@Override
	public void setPrevious(final Module... prev) {
		System.out.println("Input cannot have previous modules");
	}
}
