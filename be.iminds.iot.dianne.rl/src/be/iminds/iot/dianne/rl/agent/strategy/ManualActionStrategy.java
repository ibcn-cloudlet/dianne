package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.rl.agent.api.ManualActionController;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(property={"strategy=manual"})
public class ManualActionStrategy implements ActionStrategy, ManualActionController {

	private Tensor action;
	
	@Override
	public Tensor selectActionFromOutput(Tensor output, long i) {
		return action;
	}

	@Override
	public void setAction(Tensor a){
		this.action = a;
	}

	@Override
	public void configure(Map<String, String> config) {
	}
	
}
