package be.iminds.iot.dianne.nn.module.layer;

import be.iminds.iot.dianne.nn.module.AbstractModule;

public class Linear extends AbstractModule {

	@Override
	protected void forward() {
		System.out.println("FORWARD Linear "+id);
	}

	@Override
	protected void backward() {
		System.out.println("BACKWARD Linear "+id);
	}

}
