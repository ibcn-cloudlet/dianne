package be.iminds.iot.dianne.nn.learn.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=learn",
				  "osgi.command.function=stopLearn"},
		immediate=true)
public class DianneLearnCommands {

	private Learner learner;
	private DiannePlatform platform;
	
	public void learn(String nnName, String dataset, String tag){
		try {
			// default configuration values
			Map<String, String> config = new HashMap<String, String>();
			config.put("batchSize", "10");
			config.put("criterion", "MSE");
			config.put("learningRate", "0.1");
			config.put("momentum", "0.9");
			config.put("regularization", "0.001");
			
			if(tag!=null){
				config.put("tag", tag);
			}
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
			learner.learn(nni, dataset, config);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	public void learn(String nnName, String dataset){
		learn(nnName, dataset, null);
	}
	
	public void stopLearn(){
		this.learner.stop();
	}
	
	@Reference
	void setLearner(Learner l){
		this.learner = l;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}
