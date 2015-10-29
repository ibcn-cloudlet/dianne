package be.iminds.iot.dianne.api.coordinator;

import java.util.Map;

import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public interface DianneCoordinator {

	Promise<LearnResult> learn(String nnName, String dataset, Map<String, String> config);
	
	Promise<LearnResult> learn(NeuralNetworkDTO nn, String dataset, Map<String, String> config);

	Promise<Evaluation> eval(String nnName, String dataset, Map<String, String> config);
	
	Promise<Evaluation> eval(NeuralNetworkDTO nn, String dataset, Map<String, String> config);

}
