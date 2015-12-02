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

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.nn.learn.factory.LearnerFactory;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.unique=true"})
public class SGDLearner implements Learner {
	
	protected UUID learnerId;
	
	protected DataLogger logger;
	
	protected TensorFactory factory;
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected Thread learnerThread = null;
	protected volatile boolean learning = false;
	protected Processor processor;
	
	protected int syncInterval = 1000;
	protected boolean clean = false;
	
	// the network we are currently training
	protected NeuralNetwork nn;
	
	protected String tag = null;
	
	// initial parameters
	protected Map<UUID, Tensor> parameters = null;
	
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
	public LearnProgress getProgress(){
		return new LearnProgress(i, error);
	}
	
	@Override
	public void learn(NeuralNetworkInstanceDTO nni, String dataset,
			Map<String, String> config) throws Exception {
		if(learning){
			throw new Exception("Already running a learning session here");
		}
		
		// reset
		parameters = null;
		i = 0;
		
		// read config
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		
		if(config.containsKey("syncInterval")){
			syncInterval = Integer.parseInt(config.get("syncInterval")); 
		}
		
		if (config.containsKey("clean")){
			clean = Boolean.parseBoolean(config.get("clean"));
		}
		
		if (config.containsKey("trace")){
			trace = Boolean.parseBoolean(config.get("trace"));
		}
		
		System.out.println("Learner Configuration");
		System.out.println("=====================");
		System.out.println("* dataset = "+dataset);
		System.out.println("* tag = "+tag);
		System.out.println("* syncInterval = "+syncInterval);
		System.out.println("* clean = " +clean);
		System.out.println("---");
		
		// Fetch the dataset
		Dataset d = datasets.get(dataset);
		if(d==null){
			throw new Exception("Dataset "+dataset+" not available");
		}
		
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
		
		// initialize nn parameters
		if(clean){
			nn.randomizeParameters();
			publishParameters();
		} else {
			loadParameters();
		}
		
		// first get parameters for preprocessing?
		nn.getPreprocessors().values().stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(d);
			}
		);
		
		// create a Processor from config
		processor = LearnerFactory.createProcessor(factory, nn, d, config, logger);
		
		learnerThread = new Thread(new Runnable() {
			
			@Override
			public void run() {
				learning = true;
				float err = 0;

				do {
					nn.getTrainables().entrySet().stream().forEach(e -> {
						e.getValue().zeroDeltaParameters();
					});
					
					err = processor.processNext();
					if(i==0){
						error = err;
					} else {
						error = (1 - alpha) * error + alpha * err;
					}

					if(trace)
						System.out.println(error);
					
					nn.getTrainables().entrySet().stream().forEach(e -> {
						e.getValue().updateParameters(1.0f);
					});
						
					if(syncInterval>0){
						if(i!=0 && i % syncInterval == 0){
							// publish weights
							publishParameters();
						}
					}
					
					i++;
				} while(learning);
				System.out.println("Stopped learning");
			}
		});
		learnerThread.start();
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

	protected void loadParameters(){
		try {
			if(tag==null){
				nn.loadParameters();
			} else {
				parameters = nn.loadParameters(tag);
			}
		} catch(Exception ex){
			System.out.println("Failed to load parameters "+tag+", fill with random parameters");
			nn.randomizeParameters();
			publishParameters();
		}
	}
	
	protected void publishParameters(){
		if(parameters!=null){
			// publish delta
			if(tag==null){
				nn.storeDeltaParameters(parameters);
			} else {
				nn.storeDeltaParameters(parameters, tag);
			}
		} else {
			// just publish initial values
			if(tag==null){
				nn.storeParameters();
			} else {
				nn.storeParameters(tag);
			}
		}
		
		// fetch update again from repo (could be merged from other learners)
		loadParameters();
	}
	
	@Activate
	public void activate(BundleContext context){
		this.learnerId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	void setDataLogger(DataLogger l){
		this.logger = l;
	}
}

