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

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractLearner implements Learner {
	
	protected UUID learnerId;
	
	protected DataLogger logger;
	
	protected TensorFactory factory;
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected Thread learnerThread = null;
	protected volatile boolean learning = false;

	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected SamplingStrategy sampling;
	
	protected int syncInterval = 1000;
	protected boolean clean = false;
	
	// the network we are currently training
	protected NeuralNetwork nn;
	
	protected Dataset dataset;
	
	protected String tag = null;
	
	// initial parameters
	protected Map<UUID, Tensor> previousParameters = null;
	
	private static final float alpha = 1e-2f;
	protected float error = 0;
	protected long i = 0;
	
	protected boolean trace = false;
	
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
		if(learning){
			throw new Exception("Already running a learning session here");
		}
		
		learning = true;
		try {
			// reset
			previousParameters = null;
			i = 0;
			
			// Fetch the dataset
			loadDataset(d);
			
			// Load neural network instance(s)
			loadNNs(nni);
			
			// initialize nn parameters
			initializeParameters();
			
			// read config
			System.out.println("Learner Configuration");
			System.out.println("=====================");
			System.out.println("* dataset = "+d);
			
			loadConfig(config);

			System.out.println("---");
			
			criterion = LearnerUtil.createCriterion(factory, config);
			sampling = LearnerUtil.createSamplingStrategy(dataset, config);
			gradientProcessor = LearnerUtil.createGradientProcessor(factory, nn, dataset, config, logger);
			
			learnerThread = new Thread(() -> {
				try {
					preprocess();
					
					do {
						// process training sample(s) for this iteration
						float err = process(i);
						
						// keep track of error
						if(i==0){
							error = err;
						} else {
							error = (1 - alpha) * error + alpha * err;
						}
	
						if(trace)
							System.out.println(error);
						
						// publish parameters to repository
						publishParameters(i);
						
						i++;
					} while(learning);
				} catch(Throwable t){
					System.err.println("Error during learning");
					t.printStackTrace();
					learning = false;
				}
				System.out.println("Stopped learning");
			});
			learnerThread.start();
		
		} catch(Exception e){
			System.err.println("Failed starting learner");
			e.printStackTrace();
			learning = false;
		}	
	}
	
	@Override
	public void stop() {
		if(learning){
			learning = false;
			try {
				learnerThread.join();
			} catch (InterruptedException e) {
			}
		}
	}
	
	/**
	 * Fetch configuration parameters for this learner from the configuration map
	 */
	protected void loadConfig(Map<String, String> config){
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		System.out.println("* tag = "+tag);
		
		if(config.containsKey("syncInterval")){
			syncInterval = Integer.parseInt(config.get("syncInterval")); 
		}
		System.out.println("* syncInterval = "+syncInterval);

		if (config.containsKey("clean")){
			clean = Boolean.parseBoolean(config.get("clean"));
		}
		System.out.println("* clean = " +clean);

		
		if (config.containsKey("trace")){
			trace = Boolean.parseBoolean(config.get("trace"));
		}
		System.out.println("* trace = " +trace);
		
	}
	
	/**
	 * Load the Dataset object from the provided dataset name
	 */
	protected void loadDataset(String d){
		dataset = datasets.get(d);
		if(dataset==null){
			throw new RuntimeException("Dataset "+d+" not available");
		}
	}

	/**
	 * Load NeuralNetwork objects from provided instance dtos
	 */
	protected void loadNNs(NeuralNetworkInstanceDTO...nni) throws Exception{
		nn = dianne.getNeuralNetwork(nni[0]).getValue();
		nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
		// store the labels
		nn.setOutputLabels(dataset.getLabels());
	}
	
	/**
	 * Initialize the parameters for all neural network instances before learning starts
	 */
	protected void initializeParameters(){
		if(clean){
			resetParameters();
		} else {
			loadParameters();
		}
	}

	/**
	 * Publish parameters (or deltas ) to the repository
	 */
	protected void publishParameters(long i){
		if(syncInterval>0 && i!=0 && i % syncInterval == 0){
			// publish delta
			nn.storeDeltaParameters(previousParameters, tag);
				
			// fetch update again from repo (could be merged from other learners)
			loadParameters();
		}
	}
	
	private void resetParameters(){
		nn.randomizeParameters();
		nn.storeParameters(tag);
		previousParameters =  nn.getParameters().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null)));
	}
	
	private void loadParameters(){
		try {
			previousParameters = nn.loadParameters(tag);
		} catch(Exception ex){
			System.out.println("Failed to load parameters "+tag+", fill with random parameters");
			resetParameters();
		}
	}
	
	/**
	 * Run any preprocessing procedures before the actual learning starts
	 */
	protected void preprocess(){
		// first get parameters for preprocessing?
		nn.getPreprocessors().values().stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(dataset);
			}
		);
	}
	
	/**
	 * Actual training logic.
	 * 
	 * Each concrete Learner should update the neural network weights in this method.
	 */
	protected abstract float process(long i);
	
	@Activate
	public void activate(BundleContext context){
		this.learnerId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Reference
	protected void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Reference
	protected void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	protected void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	protected void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	protected void setDataLogger(DataLogger l){
		this.logger = l;
	}
}

