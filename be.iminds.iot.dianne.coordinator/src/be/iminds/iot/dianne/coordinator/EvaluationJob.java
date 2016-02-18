package be.iminds.iot.dianne.coordinator;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public class EvaluationJob extends AbstractJob<EvaluationResult> {

	private EvaluationResult result = new EvaluationResult();
	
	public EvaluationJob(DianneCoordinatorImpl coord,
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		super(coord, nn, d, c);
	}
	
	@Override
	public void execute() throws Exception {
		
		Map<String, String> evalConfig = new HashMap<>(config);
		if(!evalConfig.containsKey("range")){
			evalConfig.put("range", config.get("testSet"));
		}

		for(UUID target : targets){
			// TODO in parallel?
			Evaluator evaluator = coordinator.evaluators.get(target);
			// TODO gather eval results
			Evaluation e = evaluator.eval(nnis.get(target), dataset, evalConfig);
			result.evaluations.put(target, e);
		}
		
		done(result);
	}

}
