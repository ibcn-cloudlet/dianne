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
import java.util.EnumSet;
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
import be.iminds.iot.dianne.api.nn.learn.LearningStrategyFactory;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.nn.learn.config.LearnerConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service=Learner.class, 
property={"aiolos.unique=true",
		  "dianne.learner.category=EXPERIMENTAL"})
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
	private Map<UUID, NeuralNetwork> nns;
	private Dataset dataset;

	// Learning strategy
	private LearningStrategyFactory factory;
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
		if(learning)
			throw new Exception("Already running a learning session here");
		
		learning = true;
		
		try {
			// Reset
			previousParameters = new HashMap<>();
			nns = new HashMap<>();
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
			for(NeuralNetwork nn : nns.values())
				initializeParameters(nn);
			
			// TODO create LearnStrategy from factory/config?
			strategy = factory.createLearningStrategy(this.config.strategy);
			
			learnerThread = new Thread(() -> {
				try {
					strategy.setup(config, dataset, nns.values().toArray(new NeuralNetwork[nns.size()]));
					
					for(i = 0; learning; i++) {
						// Process training sample(s) for this iteration
						progress = strategy.processIteration(i);
						
						if(Float.isNaN(progress.error)){
							// if error is NaN, trigger something to repo to catch notification
							throw new Exception("Learner error became NaN");
						}
						
						if(this.config.trace)
							System.out.println("Batch: "+i+"\tError: "+progress.error);
						
						// Publish parameters to repository
						if(this.config.syncInterval > 0 && i % this.config.syncInterval == 0){
							for(NeuralNetwork nn : nns.values())
								publishParameters(nn);
						}
						
						// Publish progress
						publishProgress(progress);
					}
				} catch(Throwable t){
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
				publishDone();
				
				progress = null;
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
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
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
		for(NeuralNetworkInstanceDTO dto : nni){
			if(dto != null){
				NeuralNetwork nn = dianne.getNeuralNetwork(dto).getValue();
				nns.put(dto.id, nn);
	
				// Set module mode to blocking
				nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
				
				// Store the labels if classification dataset
				String[] labels = dataset.getLabels();
				if(labels!=null)
					nn.setOutputLabels(labels);
			}
		}
	}
	
	/**
	 * Initialize the parameters for all neural network instances before learning starts
	 */
	protected void initializeParameters(NeuralNetwork nn){
		if(config.clean)
			resetParameters(nn);
		else
			loadParameters(nn);
	}

	/**
	 * Publish parameters (or deltas ) to the repository
	 */
	protected void publishParameters(NeuralNetwork nn){
		// Publish delta
		nn.storeDeltaParameters(previousParameters.get(nn.getId()), config.tag);
				
		// Fetch update again from repo (could be merged from other learners)
		loadParameters(nn);
			
		// trigger garbage collection to clean up store tensors
		System.gc();
	}

	private void resetParameters(NeuralNetwork nn){
		// Randomize parameters
		nn.randomizeParameters();
		
		// Store new parameters
		nn.storeParameters(config.tag);
		
		// Update previous parameters
		previousParameters.put(nn.getId(), nn.getParameters().entrySet().stream().collect(
				Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null))));
	}
	
	private void loadParameters(NeuralNetwork nn){
		try {
			previousParameters.put(nn.getId(), nn.loadParameters(config.tag));
		} catch(Exception ex){
			System.out.println("Failed to load parameters "+config.tag+", fill with random parameters");
			resetParameters(nn);
		}
	}
	
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

	@Reference
	protected void setLearningStrategyFactory(LearningStrategyFactory f){
		factory = f;
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
	private void publishProgress(final LearnProgress progress){
		if(progress == null)
			return;
		
		synchronized(listenerExecutor){
			if(wait){
				try {
					listenerExecutor.wait();
				} catch (InterruptedException e) {
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
		synchronized(listenerExecutor){
			if(wait){
				try {
					listenerExecutor.wait();
				} catch (InterruptedException e) {
				}
			}
		}
		
		List<LearnerListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(LearnerListener l : copy){
			l.onException(learnerId, t.getCause()!=null ? t.getCause() : t);
		}
	}
	
	private void publishDone(){
		synchronized(listenerExecutor){
			if(wait){
				try {
					listenerExecutor.wait();
				} catch (InterruptedException e) {
				}
			}
		}
		
		List<LearnerListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(LearnerListener l : copy){
			l.onFinish(learnerId);
		}
	}
}

