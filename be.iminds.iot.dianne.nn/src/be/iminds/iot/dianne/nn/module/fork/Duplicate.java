package be.iminds.iot.dianne.nn.module.fork;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;


public class Duplicate extends Fork {

	private Tensor tempAcc = null;
	
	public Duplicate(TensorFactory factory) {
		super(factory);
	}
	
	public Duplicate(TensorFactory factory, UUID id) {
		super(factory, id);
	}

	@Override
	protected void forward() {
		// just duplicate output
		// TODO copy needed?
		for(UUID id : outputs.keySet()){
			outputs.put(id, input);
		}
	}

	@Override
	protected void backward() {
		// accumulate gradOutputs in gradInput
		if(tempAcc==null){
			tempAcc = factory.createTensor(input.dims());
		}
		tempAcc.fill(0.0f);
		for(Tensor t : gradOutputs.values()){
			tempAcc = factory.getTensorMath().add(tempAcc, tempAcc, t);
		}
		gradInput = tempAcc.copyInto(gradInput);
	}
	
}
