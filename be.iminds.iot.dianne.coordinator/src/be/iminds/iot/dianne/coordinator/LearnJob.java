package be.iminds.iot.dianne.coordinator;

import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.RepositoryListener;

public class LearnJob extends AbstractJob<LearnResult> implements RepositoryListener {

	private ServiceRegistration reg;
	
	private long maxIterations = 10000;
	
	private LearnResult result;
	
	public LearnJob(DianneCoordinatorImpl coord, 
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		super(coord, nn, d, c);
	}
		
	@Override
	public void execute() throws Exception {
		
		if(config.containsKey("maxIterations")){
			maxIterations = Long.parseLong(config.get("maxIterations"));
		}
		
		Dictionary<String, Object> props = new Hashtable();
		String[] t = nnis.values().stream().map(nni -> ":"+nni.id.toString()).collect(Collectors.toList()).toArray(new String[nnis.size()]);
		props.put("targets", t);
		props.put("aiolos.unique", true);
		reg = coordinator.context.registerService(RepositoryListener.class, this, props);
		
		// TODO deploy dataset?
		
		// set trainging set
		final Map<String, String> learnConfig = new HashMap<>(config);
		learnConfig.put("range", config.get("trainingSet"));

		System.out.println("Start Learn Job");
		System.out.println("===============");
		System.out.println("* nn: "+nn.name);
		System.out.println("* dataset: "+dataset);
		System.out.println("* maxIterations: "+maxIterations);
		System.out.println("---");
		
		// start learning on each learner
		for(UUID target : targets){
			Learner learner = coordinator.learners.get(target);
			learner.learn(dataset, learnConfig, nnis.get(target));
		}
	}

	@Override
	public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds, String... tag) {
		if(deferred.getPromise().isDone()){
			return;
		}
		
		// check stop conditition
		// TODO aggregate over all learners?
		// or make sure to check the learner this update comes from
		LearnProgress progress = coordinator.learners.get(targets.get(0)).getProgress();
		
		// maxIterations stop condition 
		boolean stop = progress.iteration >= maxIterations;
		
		// TODO other stop conditions
		// - check error rate evolution (on train and/or validationSet)
		// - stop at a certain error rate
		// - stop after certain time
		// ...
		
		// if stop ... assemble result object and resolve
		if(stop){
			// TODO keep complete history in result
			result = new LearnResult(progress.error, progress.iteration);
			done(result);
		}
	}

	@Override
	public void cleanup() {
		reg.unregister();
		
		for(UUID target : targets){
			Learner learner = coordinator.learners.get(target);
			if(learner!=null){
				learner.stop();
			}
		}
	}

	
}
