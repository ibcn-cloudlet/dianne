package be.iminds.iot.dianne.nn.module.activation;

import be.iminds.iot.dianne.nn.module.AbstractModule;

public class Tanh extends AbstractModule {

	@Override
	protected void forward() {
		System.out.println("FORWARD Tanh "+id);
	}

	@Override
	protected void backward() {
		System.out.println("BACKWARD Tanh "+id);
	}

}
