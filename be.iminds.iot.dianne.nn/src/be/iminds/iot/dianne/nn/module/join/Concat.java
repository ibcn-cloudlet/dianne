package be.iminds.iot.dianne.nn.module.join;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.TensorFactory;

public class Concat extends Join {

	public Concat(TensorFactory factory) {
		super(factory);
	}
	
	public Concat(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		// concat from N equal parts in dimension 1
		// TODO other split strategies?
		if(prev!=null){
			int[] dims = inputs.values().iterator().next().dims();
			int size = dims[0];
			if(output==null){
				dims[0] = dims[0]*inputs.size();
				output = factory.createTensor(dims);
			}
		
			for(int i=0;i<prev.length;i++){
				inputs.get(prevIds[i]).copyInto(output.narrow(0, i*size, size));
			}
		}
	}

	@Override
	protected void backward() {
		if(prev!=null){
			int size = inputs.values().iterator().next().size(0);
			for(int i=0;i<prev.length;i++){
				gradInputs.put(prevIds[i], gradOutput.narrow(0, i*size, size));
			}
		}
	}

}
