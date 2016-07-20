/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.LearnerListener;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.nn.learn.config.LearnerConfig;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractLearner implements Learner {
	
	// Listeners
	private static ExecutorService listenerExecutor = Executors.newSingleThreadExecutor(); 
	protected List<LearnerListener> listeners = Collections.synchronizedList(new ArrayList<>());


	// Identification
	protected UUID learnerId;
	
	// References
	protected DataLogger logger;
	protected Dianne dianne;
	protected DianneDatasets datasets;
	
	// Threading
	protected Thread learnerThread;
	protected volatile boolean learning = false;
	
	// Learning procedure
	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected SamplingStrategy sampling;
	
	// Network and data
	protected NeuralNetwork nn;
	protected Dataset dataset;
	
	// Configuration
	protected LearnerConfig config;
	
	// Initial  parameters
	protected Map<UUID, Tensor> previousParameters;
	
	// Training progress
	protected volatile float error = 0;
	protected volatile long i = 0;
	
	// Fixed properties
	protected static final float alpha = 1e-2f;

	// Retry with previously stored parameters in case of NaN
	protected int nanretry = 0;
	
	
	@Override
	public UUID getLearnerId(){
		return learnerId;
	}
	
	@Override
	public boolean isBusy(){
		return learning;
	}
	
	@Override
	public LearnProgress getProgress() {
		if(!learning)
			return null;
		return new LearnProgress(i, error);
	}
	
	@Override
	public void learn(String d, Map<String, String> config, NeuralNetworkInstanceDTO... nni) throws Exception {
		if(learning)
			throw new Exception("Already running a learning session here");
		
		learning = true;
		
		try {
			// Reset
			previousParameters = null;
			i = 0;
			
			// Read config
			System.out.println("Learner Configuration");
			System.out.println("=====================");
			
			loadConfig(config);
			
			// Fetch the dataset
			loadDataset(d, config);
			
			// Load neural network instance(s)
			loadNNs(nni);

			// Initialize NN parameters
			initializeParameters();
			
			// setup criterion, sampling strategy and gradient processor
			sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);

			criterion = CriterionFactory.createCriterion(this.config.criterion);
			gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config, logger);
			
			nanretry = this.config.NaNretry;
			
			learnerThread = new Thread(() -> {
				try {
					preprocess(d, config);
					
					for(i = 0; learning; i++) {
						// Process training sample(s) for this iteration
						float err = process(i);
						
						if(Float.isNaN(err)){
							if(nanretry != 0){
								System.out.println("Retry after NaN");
								loadParameters();
								if(nanretry > 0)
									nanretry--;
								continue;
							} else {
								throw new Exception("Learner error became NaN");
							}
						}
						
						// Keep track of error
						if(i==0)
							error = err;
						else
							error = (1 - alpha) * error + alpha * err;
	
						if(this.config.trace)
							System.out.println("Batch: "+i+"\tError: "+error);
						
						// Publish parameters to repository
						publishParameters(i);
						
						// Publish progress
						publishProgress(i);
					}
				} catch(Throwable t){
					learning = false;

					System.err.println("Error during learning");
					t.printStackTrace();
					
					List<LearnerListener> copy = new ArrayList<>();
					synchronized(listeners){
						copy.addAll(listeners);
					}
					for(LearnerListener l : copy){
						l.onException(learnerId, t.getCause()!=null ? t.getCause() : t);
					}
					
					return;
				} finally {
					datasets.releaseDataset(dataset);
					System.gc();
				}

				System.out.println("Stopped learning");
				List<LearnerListener> copy = new ArrayList<>();
				synchronized(listeners){
					copy.addAll(listeners);
				}
				for(LearnerListener l : copy){
					l.onFinish(learnerId, new LearnProgress(i, error));
				}
			});
			learnerThread.start();
		
		} catch(Exception e){
			System.err.println("Failed starting learner");
			e.printStackTrace();
			learning = false;
			throw e;
		}	
	}
	
	@Override
	public void stop() {
		if(learning){
			learning = false;
			
			try {
				learnerThread.join();
			} catch (InterruptedException e) {}
		}
	}
	
	/**
	 * Fetch configuration parameters for this learner from the configuration map
	 */
	protected void loadConfig(Map<String, String> config){
		this.config = DianneConfigHandler.getConfig(config, LearnerConfig.class);
	}
	
	/**
	 * Load the Dataset object from the provided dataset name
	 */
	protected void loadDataset(String d, Map<String, String> config){
		dataset = datasets.configureDataset(d, config);
		
		if(dataset==null)
			throw new RuntimeException("Dataset "+d+" not available");
	}

	/**
	 * Load NeuralNetwork objects from provided instance dtos
	 */
	protected void loadNNs(NeuralNetworkInstanceDTO...nni) throws Exception {
		// Get the reference
		nn = dianne.getNeuralNetwork(nni[0]).getValue();
		
		// Store the labels if classification dataset
		String[] labels = dataset.getLabels();
		if(labels!=null)
			nn.setOutputLabels(labels);
	}
	
	/**
	 * Initialize the parameters for all neural network instances before learning starts
	 */
	protected void initializeParameters(){
		if(config.clean)
			resetParameters();
		else
			loadParameters();
	}

	/**
	 * Publish parameters (or deltas ) to the repository
	 */
	protected void publishParameters(long i){
		if(config.syncInterval > 0 && i % config.syncInterval==0 && i!=0){
			// Publish delta
			nn.storeDeltaParameters(previousParameters, config.tag);
				
			// Fetch update again from repo (could be merged from other learners)
			loadParameters();
			
			// trigger garbage collection to clean up store tensors
			System.gc();
		}
	}

	private void resetParameters(){
		// Randomize parameters
		nn.randomizeParameters();
		
		// Store new parameters
		nn.storeParameters(config.tag);
		
		// Update previous parameters
		previousParameters = nn.getParameters().entrySet().stream().collect(
				Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null)));
	}
	
	private void loadParameters(){
		try {
			previousParameters = nn.loadParameters(config.tag);
		} catch(Exception ex){
			System.out.println("Failed to load parameters "+config.tag+", fill with random parameters");
			resetParameters();
		}
	}
	
	/**
	 * Run any preprocessing procedures before the actual learning starts
	 */
	protected void preprocess(String d, Map<String, String> c){
		if(!nn.getPreprocessors().values().stream()
			.filter(p -> !p.isPreprocessed())
			.findFirst().isPresent())
			return;
		
		// preprocess on the train set without
		System.out.println("Preprocess!");
		HashMap<String, String> trainSetConfig = new HashMap<>();
		if(c.containsKey("range"))
			trainSetConfig.put("range", c.get("range"));
		
		Dataset preprocessSet = datasets.configureDataset(d, trainSetConfig);
		
		try {
			// TODO first get parameters for preprocessing?
			nn.getPreprocessors().values().stream()
				.filter(p -> !p.isPreprocessed())
				.forEach(p -> p.preprocess(preprocessSet));
			
			Map<UUID, Tensor> preprocessorParameters = new HashMap<>();
			nn.getPreprocessors().entrySet().stream().forEach(e -> preprocessorParameters.put(e.getKey(), e.getValue().getParameters()));
			nn.storeParameters(preprocessorParameters, config.tag);
		} finally {
			datasets.releaseDataset(preprocessSet);
		}
	}
	
	/**
	 * Actual training logic.
	 * 
	 * Each concrete Learner should update the neural network weights in this method.
	 */
	protected abstract float process(long i) throws Exception;
	
	@Activate
	public void activate(BundleContext context){
		this.learnerId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Reference
	protected void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	protected void setDianneDatasets(DianneDatasets d){
		datasets = d;
	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	protected void setDataLogger(DataLogger l){
		this.logger = l;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	protected void addListener(LearnerListener listener, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			boolean listen = false;
			for(String target : targets){
				if(learnerId.toString().equals(target)){
					listen  = true;
				}
			}
			if(!listen)
				return;	
		}
		this.listeners.add(listener);
	}
	
	protected void removeListener(LearnerListener listener, Map<String, Object> properties){
		this.listeners.remove(listener);
	}
	
	/**
	 * Publish progress on sync interval times
	 */
	private void publishProgress(long i){
		if(config.syncInterval > 0 && i % config.syncInterval == 0){
			final LearnProgress progress =  getProgress();
			if(progress == null)
				return;
			
			listenerExecutor.submit(()->{
				List<LearnerListener> copy = new ArrayList<>();
				synchronized(listeners){
					copy.addAll(listeners);
				}
				for(LearnerListener l : copy){
					l.onProgress(learnerId, progress);
				}
			});
		}
	}
}

