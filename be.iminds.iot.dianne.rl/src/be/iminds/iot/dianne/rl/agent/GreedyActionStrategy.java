package be.iminds.iot.dianne.rl.agent;

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"strategy=greedy"})
public class GreedyActionStrategy implements ActionStrategy {
	
	private TensorFactory factory;
	
	private double epsilon = 1e0;
	private double decay = 1e-6;
	
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
	
	@Override
	public void configure(Map<String, String> config) {
		if (config.containsKey("epsilon"))
			epsilon = Double.parseDouble(config.get("epsilon"));
		
		if (config.containsKey("decay"))
			decay = Double.parseDouble(config.get("decay"));
	}

	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory =f;
	}
}
