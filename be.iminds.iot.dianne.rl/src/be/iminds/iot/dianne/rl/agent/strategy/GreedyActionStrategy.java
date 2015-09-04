package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"strategy=greedy"})
public class GreedyActionStrategy implements ActionStrategy {
	
	private TensorFactory factory;
	
	private double epsilonMax = 1e0;
	private double epsilonMin = 0;
	private double epsilonDecay = 1e-6;
	
	private DataLogger logger = null;
	private String[] loglabels = new String[]{"Q0", "Q1", "Q2", "epsilon"};
	
	public Tensor selectActionFromOutput(Tensor output, long i) {
		
		Tensor action = factory.createTensor(output.size());
		action.fill(-1);
		
		double epsilon = epsilonMin + (epsilonMax - epsilonMin) * Math.exp(-i * epsilonDecay);
		
		if(logger!=null){
			logger.log("AGENT", loglabels, output.get()[0],output.get()[1],output.get()[2], (float)epsilon);
		}
		
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
		
		System.out.println("Greedy Action Selection");
		System.out.println("* epsilon max = "+epsilonMax);
		System.out.println("* epsilon min = "+epsilonMin);
		System.out.println("* epsilon decay = "+epsilonDecay);
		System.out.println("---");
	}

	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory =f;
	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	public void setDataLogger(DataLogger l){
		this.logger = l;
		this.logger.setAlpha("epsilon", 1f);
		this.logger.setAlpha("Q0", 1f);
		this.logger.setAlpha("Q1", 1f);
		this.logger.setAlpha("Q2", 1f);
	}

}
