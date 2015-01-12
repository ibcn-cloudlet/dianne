package be.iminds.iot.dianne.nn.module.io;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.tensor.Tensor;

public class Output extends AbstractModule {

	public void expected(Tensor e){
		backward(this.id, e);
	}
	
	public Tensor getOutput(){
		return output;
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
	public void setNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}

}
