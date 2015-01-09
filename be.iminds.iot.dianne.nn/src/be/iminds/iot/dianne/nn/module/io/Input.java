package be.iminds.iot.dianne.nn.module.io;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;

public class Input extends AbstractModule {

	public void input(Tensor input){
		forward(this.id, input);
	}
	
	@Override
	protected void forward() {
		System.out.println("FORWARD INPUT "+id);
		output = input;
	}

	@Override
	protected void backward() {
		System.out.println("RESULT: "+gradOutput);
		gradInput = gradOutput;
	}

	@Override
	public void setPrevious(final Module... prev) {
		System.out.println("Input cannot have previous modules");
	}

}
