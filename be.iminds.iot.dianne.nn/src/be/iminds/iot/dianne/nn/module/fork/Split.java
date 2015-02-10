package be.iminds.iot.dianne.nn.module.fork;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.TensorFactory;

public class Split extends Fork {

	public Split(TensorFactory factory) {
		super(factory);
	}
	
	public Split(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		// split in N equal parts in dimension 1
		// TODO other split strategies?
		if(next!=null){
			int size = input.size(0)/next.length;
			for(int i=0;i<next.length;i++){
				outputs.put(nextIds[i], input.narrow(0, i*size, size));
			}
		}
	}

	@Override
	protected void backward() {
		if(next!=null){
			int[] dims = gradOutputs.values().iterator().next().dims();
			int size = dims[0];
			if(output==null){
				dims[0] = dims[0]*gradOutputs.size();
				gradInput = factory.createTensor(dims);
			}

			for(int i=0;i<next.length;i++){
				gradOutputs.get(nextIds[i]).copyInto(gradInput.narrow(0, i*size, size));
			}
		}
		
	}

}
