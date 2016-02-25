package be.iminds.iot.dianne.coordinator;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public class EvaluationJob extends AbstractJob<EvaluationResult> {

	private EvaluationResult result = null;
	private Map<UUID, Evaluation> results = new HashMap<>();

	public EvaluationJob(DianneCoordinatorImpl coord,
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		super(coord, Type.EVALUATE, nn, d, c);
	}
	
	@Override
	public void execute() throws Exception {
		
		Map<String, String> evalConfig = new HashMap<>(config);
		if(!evalConfig.containsKey("range")){
			evalConfig.put("range", config.get("testSet"));
		}

		Thread[] threads = new Thread[targets.size()];
		for(int i=0;i<targets.size();i++){
			final UUID target = targets.get(i);
			threads[i] = new Thread(new Runnable(){
				public void run(){
					try {
						Evaluator evaluator = coordinator.evaluators.get(target);
						Evaluation e = evaluator.eval(nnis.get(target), dataset, evalConfig);
						results.put(target, e);
					} catch(Exception e){
						// TODO should this fail the entire job?
						results.put(target, null);
					}
				}
			});
			threads[i].start();
		}
		
		for(Thread t : threads){
			t.join();
		}
		
		result = new EvaluationResult(results);
		
		done(result);
	}

	@Override
	public EvaluationResult getProgress() {
		if(result!=null)
			return result;
		
		Map<UUID, Evaluation> progresses = new HashMap<>();
		for(UUID target : targets){
			EvaluationProgress p = null; 
			if(results.containsKey(target)){
				Evaluation eval = results.get(target);
				p = new EvaluationProgress(eval.getTotal(), eval.getTotal(), eval.evaluationTime());
			} else {
				Evaluator evaluator = coordinator.evaluators.get(target);
				p = evaluator.getProgress();
			}
			progresses.put(target, p);
		}
		return new EvaluationResult(progresses);
	}

}
