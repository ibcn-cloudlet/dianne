package be.iminds.iot.dianne.rl.agent;

import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class GreedyActionStrategy implements ActionStrategy {
	
	private TensorFactory factory;
	
	private double epsilon;
	private double decay;
	
	public GreedyActionStrategy(TensorFactory f, double epsilon, double decay){
		this.factory = f;
		this.epsilon = epsilon;
		this.decay = decay;
	}
	
	public Tensor selectActionFromOutput(Tensor output, long i) {
		Tensor action = factory.createTensor(output.size());
		action.fill(-1);

		if (Math.random() < epsilon * Math.exp(-i * decay)) {
			action.set(1, (int) (Math.random() * action.size()));
		} else {
			action.set(1, factory.getTensorMath().argmax(output));
		}
		
		return action;
	}
	
	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory =f;
	}

}
