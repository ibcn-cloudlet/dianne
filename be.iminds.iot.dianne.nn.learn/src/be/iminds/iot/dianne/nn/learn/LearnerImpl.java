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
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.LearnerListener;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.util.StrategyFactory;
import be.iminds.iot.dianne.nn.learn.config.LearnerConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(property={"aiolos.unique=true"})
public class LearnerImpl implements Learner {
	
	// Listeners
	private ExecutorService listenerExecutor = Executors.newSingleThreadExecutor(); 
	private List<LearnerListener> listeners = Collections.synchronizedList(new ArrayList<>());
	private volatile boolean wait = false;

	// Identification
	private UUID learnerId;
	
	// References
	private Dianne dianne;
	private DianneDatasets datasets;
	
	// Threading
	private Thread learnerThread;
	private volatile boolean learning = false;
	
	// Network and data
	private NeuralNetwork[] nns;
	private Dataset dataset;

	// Learning strategy
	private StrategyFactory<LearningStrategy> factory;
	private LearningStrategy strategy;
	
	// Config
	private LearnerConfig config;
	
	// Previous  parameters
	private Map<UUID, Map<UUID, Tensor>> previousParameters;
	
	// Training progress
	private volatile long i = 0;
	private LearnProgress progress;

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
		return progress;
	}
	
	@Override
	public void learn(String d, Map<String, String> config, NeuralNetworkInstanceDTO... nni) throws Exception {
		synchronized(this){
			if(learning)
				throw new Exception("Already running a learning session here");
			
			learning = true;
		}
		
		try {
			// Reset
			previousParameters = new HashMap<>();
			nns = new NeuralNetwork[nni.length];
			i = 0;
			
			// Read config
			System.out.println("Learner Configuration");
			System.out.println("=====================");
			
			this.config = DianneConfigHandler.getConfig(config, LearnerConfig.class);
			
			
			// Fetch the dataset
			dataset = datasets.configureDataset(d, config);
			
			if(dataset==null)
				throw new RuntimeException("Dataset "+d+" not available");
			
			
			// Load neural network instance(s)
			System.out.println("Neural Network(s)");
			System.out.println("---");
			int n = 0;
			for(NeuralNetworkInstanceDTO dto : nni){
				if(dto != null){
					NeuralNetwork nn = dianne.getNeuralNetwork(dto).getValue();
					nns[n++] = nn;

					// mark modules as fixed
					if(this.config.fixed.length > 0){
						Map<UUID, Trainable> tt = nn.getTrainables();
						for(UUID fixed : this.config.fixed){
							Trainable t = tt.get(fixed);
							if(t!=null){
								t.setFixed(true);
							}
						}
					}
					
					System.out.println("* "+dto.name);
				}
			}
			System.out.println("---");

			// Initialize NN parameters
			if(this.config.initTag != null){
				// load parameters from init tag and store as tag
				for(NeuralNetwork nn : nns){
					loadParameters(nn, this.config.initTag);
					nn.storeParameters(this.config.tag);
				}
			}
			
			for(NeuralNetwork nn : nns){
				initializeParameters(nn);
			}
			
			// Create learning strategy
			strategy = factory.create(this.config.strategy);
			
			if(strategy == null)
				throw new RuntimeException("LearningStrategy "+this.config.strategy+" not available");
			
			learnerThread = new Thread(() -> {
				try {
					// Trigger preprocess on NN instances before the actual training starts
					for(NeuralNetwork nn : nns){
						preprocess(nn, d, config);
					}
					
					// Setup LearningStrategy
					strategy.setup(config, dataset, nns);
					// this allows the strategy to adapt config in setup
					this.config = DianneConfigHandler.getConfig(config, LearnerConfig.class);
					
					// Actual training loop
					for(i = 0; learning; i++) {
						// Process training sample(s) for this iteration
						progress = strategy.processIteration(i);

						// Check for NaN
						if(Float.isNaN(progress.minibatchLoss)){
							throw new Exception("Learner error became NaN");
						}
						
						if(i % this.config.traceInterval == 0){
							if(this.config.trace)
								System.out.println(progress);
							
							publishProgress(progress);
						}
						
						// Publish parameters to repository
						for(int k=0;k<nns.length;k++){
							int syncInterval = (k < this.config.syncInterval.length) ? this.config.syncInterval[k] : this.config.syncInterval[0];
							if(syncInterval > 0 && i > 0 && i % syncInterval == 0){
								publishParameters(nns[k]);
							}
						}
						
						// Store intermediate parameters
						for(int k=0;k<nns.length;k++){
							int storeInterval = (k < this.config.storeInterval.length) ? this.config.storeInterval[k] : this.config.storeInterval[0];
							if(storeInterval > 0 && i > 0 && i % storeInterval == 0){
								nns[k].storeParameters(this.config.tag, ""+i);
							}
						}
						
					}
				} catch(InterruptedException e){ 
					// ignore, just interrupt the thread
				} catch(Throwable t){
					if(t.getCause() != null & t.getCause() instanceof InterruptedException){
						// ignore if this cause is from interrupt
						return;
					}
					
					learning = false;

					System.err.println("Error during learning");
					t.printStackTrace();
					
					publishError(t);
					
					return;
				} finally {
					datasets.releaseDataset(dataset);
					System.gc();
				}

				System.out.println("Stopped learning");
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
		if(!learning)
			return;
		
		synchronized(this){
			if(learning){
				learning = false;
				
				try {
					// interrupt in case of waiting during publish progress
					learnerThread.interrupt();
					learnerThread.join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				
				publishDone();
				progress = null;
			}
		}
	}
	
	/**
	 * Trigger preprocess on the NNs preprocess modules 
	 */
	private void preprocess(NeuralNetwork nn, String dataset, Map<String, String> c){
		if(!nn.getPreprocessors().values().stream()
			.filter(p -> !p.isPreprocessed())
			.findFirst().isPresent())
			return;
		
		// preprocess on the train set without
		System.out.println("Preprocess!");
		HashMap<String, String> trainSetConfig = new HashMap<>();
		if(c.containsKey("range"))
			trainSetConfig.put("range", c.get("range"));
		
		Dataset preprocessSet = datasets.configureDataset(dataset, trainSetConfig);
		
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
	 * Initialize the parameters for all neural network instances before learning starts
	 */
	private void initializeParameters(NeuralNetwork nn){
		try {
			// makes sure that fixed parameters are loaded anyhow
			loadParameters(nn, config.tag);
			
			if(config.clean){
				// if clean, randomize parameters
				resetParameters(nn);
			}
		} catch(Exception e){
			System.out.println("Failed to load parameters "+config.tag+", fill with random parameters");
			resetParameters(nn);
		}
	}

	/**
	 * Publish parameters (or deltas ) to the repository
	 */
	private void publishParameters(NeuralNetwork nn){
		// Publish delta
		nn.storeDeltaParameters(previousParameters.get(nn.getId()), config.tag);
				
		// Fetch update again from repo (could be merged from other learners)
		try {
			loadParameters(nn, config.tag);
		} catch(Exception e){
			System.out.println("Failed to load parameters after publish?!");
			e.printStackTrace();
		}
	}

	/**
	 * Reset Neural Network parameters to random initialization
	 */
	private void resetParameters(NeuralNetwork nn){
		// Randomize parameters
		nn.randomizeParameters();
		
		// Store new parameters
		nn.storeParameters(config.tag);
		
		// Update previous parameters
		previousParameters.put(nn.getId(), nn.getParameters().entrySet().stream().collect(
				Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null))));
	}
	
	/**
	 * Load parameters from the repository and store in previousParameters
	 */
	private void loadParameters(NeuralNetwork nn, String tag) throws Exception {
		previousParameters.put(nn.getId(), nn.loadParameters(tag));
		
		// TODO should this be handled somewhere else?
		
		// in case some modules are missing ... initialize those separately?!
		Map<UUID, Tensor> prev = previousParameters.get(nn.getId());
		if(prev.size() != nn.getTrainables().size()){
			for(UUID moduleId : nn.getTrainables().keySet()){
				if(!prev.containsKey(moduleId)){
					nn.randomizeParameters(moduleId);
				}
			}
			
			// store and update previous again
			nn.storeParameters(tag);
			previousParameters.put(nn.getId(), nn.getParameters().entrySet().stream().collect(
					Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null))));
		}
	}
	
	@Activate
	void activate(BundleContext context){
		this.learnerId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
	}

	@Reference
	void setLearningStrategyFactory(StrategyFactory<LearningStrategy> f){
		factory = f;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addListener(LearnerListener listener, Map<String, Object> properties){
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
	
	void removeListener(LearnerListener listener, Map<String, Object> properties){
		this.listeners.remove(listener);
	}
	
	/**
	 * Publish progress
	 */
	private void publishProgress(final LearnProgress progress){
		if(!learning)
			return;
		
		synchronized(listenerExecutor){
			if(wait){
				try {
					listenerExecutor.wait();
				} catch (InterruptedException e) {
					wait = false;
					return;
				}
			}
			wait = true;
		}
		
		listenerExecutor.submit(()->{
			List<LearnerListener> copy = new ArrayList<>();
			synchronized(listeners){
				copy.addAll(listeners);
			}
			for(LearnerListener l : copy){
				l.onProgress(learnerId, progress);
			}
			
			synchronized(listenerExecutor){
				wait = false;
				listenerExecutor.notifyAll();
			}
		});
	}
	
	private void publishError(final Throwable t){
		List<LearnerListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(LearnerListener l : copy){
			l.onException(learnerId, t.getCause()!=null ? t.getCause() : t);
		}
	}
	
	private void publishDone(){
		List<LearnerListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(LearnerListener l : copy){
			l.onFinish(learnerId);
		}
	}
}

