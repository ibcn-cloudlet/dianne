package be.iminds.iot.dianne.coordinator;

import java.util.Collection;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.Map;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;

@Component
public class DianneCoordinatorImpl implements DianneCoordinator {

	private BundleContext context;

	private DiannePlatform platform;
	private DianneRepository repository;
	// TODO for now only one learner and evaluator
	private Learner learner;
	private Evaluator evaluator;
	//private Map<UUID, Learner> learners = Collections.synchronizedMap(new HashMap<>());
	//private Map<UUID, Evaluator> evaluators = Collections.synchronizedMap(new HashMap<>());
	
	private Queue<LearnJob> queue = new LinkedBlockingQueue<>();
	
	private ExecutorService pool = Executors.newCachedThreadPool();
	
	@Override
	public Promise<LearnResult> learn(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {

		LearnJob job = new LearnJob(nn, config, dataset);
		queue.add(job);
		next();
		
		return job.getPromise();
	}

	@Override
	public Promise<LearnResult> learn(String nnName, String dataset, Map<String, String> config) {
		return learn(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	// try to do next job
	private void next(){
		// TODO one should keep a separate list of available learners
		// when calling start the learner is not yet busy...
		if(!learner.isBusy()){
			LearnJob job = queue.poll();
			if(job!=null){
				job.start(learner);
			}
		}
	}
	
	private class LearnJob implements RepositoryListener, Runnable {
		
		private Deferred<LearnResult> deferred = new Deferred<>();
		
		private NeuralNetworkDTO nn;
		private Map<String, String> config;
		private String dataset;
		
		private Learner learner;
		private NeuralNetworkInstanceDTO nni;
		
		private ServiceRegistration<RepositoryListener> reg;
		
		public LearnJob(NeuralNetworkDTO nn, Map<String, String> config, String dataset){
			this.nn = nn;
			this.config = config;
			this.dataset = dataset;
		}
		
		Promise<LearnResult> getPromise(){
			return deferred.getPromise();
		}
		
		void start(Learner learner){
			this.learner = learner;
			pool.execute(this);
		}
		
		public void run(){
			System.out.println("Executing learn job!");
	
			// do the actual Learning
			try {
				// deploy nn
				nni = platform.deployNeuralNetwork(nn.name, "Dianne Coordinator LearnJob", learner.getLearnerId());
				
				// register RepositoryListener
				Dictionary<String, Object> props = new Hashtable();
				props.put("targets", new String[]{":"+nni.id.toString()});
				props.put("aiolos.unique", true);
				reg = context.registerService(RepositoryListener.class, this, props);
				
				// TODO provide dataset?
				
				// start learning
				learner.learn(nni, dataset, config);
				
				// TODO timeout?
			
			} catch(Exception e){
				System.out.println("Job failed");
				e.printStackTrace();
				
				deferred.fail(e);
				done();
			}
		}

		@Override
		public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds,
				String... tag) {
			// check stop conditition
			LearnProgress progress = learner.getProgress();
			
			// TODO stop condition? - Evaluate on validation set?
			boolean stop = progress.iteration > 10000;
			
			// if stop ... assemble result object and resolve
			if(stop){
				try {
					Evaluation eval = evaluator.eval(nni, dataset, config);
					// TODO evaluate time on other/multiple platforms?
					LearnResult result = new LearnResult(eval.accuracy(), eval.forwardTime());
					deferred.resolve(result);
				} catch(Exception e){
					deferred.fail(e);
				} finally {
					done();
				}
			}
		}
		
		private void done(){
			if(reg!=null)
				reg.unregister();
			
			if(nni!=null)
				platform.undeployNeuralNetwork(nni);
			
			next();
		}
	}

	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform platform){
		this.platform = platform;
	}
	
	@Reference
	void setDianneRepository(DianneRepository repository){
		this.repository = repository;
	}
	
	@Reference
	void setLearner(Learner learner){
		this.learner = learner;
	}
	
	@Reference
	void setEvaluator(Evaluator evaluator){
		this.evaluator = evaluator;
	}
	
}
