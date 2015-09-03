package be.iminds.iot.dianne.rl.agent;

import java.util.Arrays;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"strategy=greedy"})
public class GreedyActionStrategy implements ActionStrategy {
	
	private TensorFactory factory;
	
	private double epsilonMax = 1e0;
	private double epsilonMin = 0;
	private double epsilonDecay = 1e-6;
	
	public Tensor selectActionFromOutput(Tensor output, long i) {
		
		Tensor action = factory.createTensor(output.size());
		action.fill(-1);
		
		double epsilon = epsilonMin + (epsilonMax - epsilonMin) * Math.exp(-i * epsilonDecay);
		
		if(i % 1000 == 0)
			System.out.println(i + "\tQ: " + Arrays.toString(output.get()) + "\te: " + epsilon);

		if (Math.random() < epsilon) {
			action.set(1, (int) (Math.random() * action.size()));
		} else {
			action.set(1, factory.getTensorMath().argmax(output));
		}
		
		return action;
	}
	
	@Override
	public void configure(Map<String, String> config) {
		if (config.containsKey("epsilonMax"))
			epsilonMax = Double.parseDouble(config.get("epsilonMax"));
		
		if (config.containsKey("epsilonMin"))
			epsilonMin = Double.parseDouble(config.get("epsilonMin"));
		
		if (config.containsKey("epsilonDecay"))
			epsilonDecay = Double.parseDouble(config.get("epsilonDecay"));
	}

	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory =f;
	}
}
