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
		System.out.println("RESULT: ");
		for(Tensor t : inputs.values()){
			System.out.println(t);
		}
	}

	@Override
	protected void backward() {
		System.out.println("BACKWARD OUTPUT");
	}
	
	@Override
	public void addNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}

	@Override
	public void removeNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}
}
