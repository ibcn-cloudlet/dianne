package be.iminds.iot.dianne.nn.eval.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=eval"},
		immediate=true)
public class DianneEvalCommands {

	private Evaluator evaluator;
	private DiannePlatform platform;
	
	public void eval(String nnName, String dataset, String... properties){
		try {
			Map<String, String> config = createEvalConfig(properties);
			
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
			Evaluation e = evaluator.eval(nni, dataset, config);
			System.out.println("Accuracy "+e.accuracy());
			System.out.println("Sample time "+e.getSampleTime()+" ms");
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	private Map<String, String> createEvalConfig(String[] properties){
		Map<String, String> config = new HashMap<String, String>();
		
		for(String property : properties){
			String[] p = property.split("=");
			if(p.length==2){
				config.put(p[0].trim(), p[1].trim());
			}
		}
		
		return config;
	}
	
	
	@Reference
	void setEvaluator(Evaluator e){
		this.evaluator = e;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}
