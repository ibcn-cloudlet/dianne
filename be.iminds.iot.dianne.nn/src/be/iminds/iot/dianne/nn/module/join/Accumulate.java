package be.iminds.iot.dianne.nn.module.join;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;


public class Accumulate extends Join {

	public Accumulate(TensorFactory factory) {
		super(factory);
	}
	
	public Accumulate(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		// accumulate inputs
		if(output==null){
			output = factory.createTensor(inputs.values().iterator().next().dims());
		}
		output.fill(0.0f);
		for(Tensor t : inputs.values()){
			if(t!=null)
				output = factory.getTensorMath().add(output, output, t);
		}
	}

	@Override
	protected void backward() {
		// forward same error to all
		// TODO copy needed?
		for(UUID id : gradInputs.keySet()){
			gradInputs.put(id, gradOutput);
		}
	}

}
