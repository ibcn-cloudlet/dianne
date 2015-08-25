package be.iminds.iot.dianne.rl.learn;

import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.learn.Learner;

@Component
public class DeepQLearner implements Learner {

	@Override
	public void learn(String nnName, String dataset, Map<String, String> config) throws Exception {
		// TODO Auto-generated method stub

	}

	@Override
	public void stop() {
		// TODO Auto-generated method stub

	}

}
