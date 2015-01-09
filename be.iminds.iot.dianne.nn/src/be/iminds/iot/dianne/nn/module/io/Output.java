package be.iminds.iot.dianne.nn.module.io;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;

public class Output extends AbstractModule {

	public void expected(Tensor e){
		backward(this.id, e);
	}
	
	@Override
	protected void forward() {
		System.out.println("RESULT: "+input);
		output = input;
	}

	@Override
	protected void backward() {
		System.out.println("BACKWARD OUTPUT "+id);
		gradInput = gradOutput;
	}
	
	@Override
	public void setNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}

}
