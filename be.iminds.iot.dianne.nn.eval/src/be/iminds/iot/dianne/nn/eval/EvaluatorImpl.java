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
package be.iminds.iot.dianne.nn.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.EvaluationStrategy;
import be.iminds.iot.dianne.api.nn.eval.EvaluationStrategyFactory;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.eval.EvaluatorListener;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.util.StrategyFactory;
import be.iminds.iot.dianne.nn.eval.config.EvaluatorConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

/**
 * The AbstractEvaluator has all mechanics to loop through (part of) a Dataset
 * and forward each sample.
 * 
 * Concrete evaluators should implement the evalOutput(index, output, expected output)
 * and update the error/confusionMatrix
 * @author tverbele
 *
 */
@Component(property={"aiolos.unique=true"})
public class EvaluatorImpl implements Evaluator {
	
	private ExecutorService listenerExecutor = Executors.newSingleThreadExecutor(); 
	private List<EvaluatorListener> listeners = Collections.synchronizedList(new ArrayList<>());
	
	private UUID evaluatorId;
	
	private Dianne dianne;
	private DianneDatasets datasets;
	
	private EvaluatorConfig config;
	
	private volatile boolean evaluating = false;

	private StrategyFactory<EvaluationStrategy> factory;
	private EvaluationStrategy strategy;
	private EvaluationProgress progress;
	
	private long tStart, tEnd;
	
	@Override
	public UUID getEvaluatorId(){
		return evaluatorId;
	}

	@Override
	public synchronized Evaluation eval(String dataset,
			Map<String, String> config, NeuralNetworkInstanceDTO... nni) throws Exception {
		if(evaluating){
			throw new Exception("Already running an evaluation session here");
		}
		evaluating = true;
		
		Dataset d = null;
		try {
			System.out.println("Evaluator Configuration");
			System.out.println("=======================");
			
			this.config = DianneConfigHandler.getConfig(config, EvaluatorConfig.class);

			// Fetch the dataset
			d = datasets.configureDataset(dataset, config);
			if(d==null)
				throw new Exception("Dataset "+dataset+" not available");

			// Fetch Neural Network instances and load parameters
			NeuralNetwork[] nns = new NeuralNetwork[nni.length];
			int n = 0;
			for(NeuralNetworkInstanceDTO dto : nni){
				if(dto != null){
					NeuralNetwork nn = dianne.getNeuralNetwork(dto).getValue();
					nns[n++] = nn;
					
					try {
						if(this.config.tag==null){
							nn.loadParameters();
						} else {
							nn.loadParameters(this.config.tag);
						}
					} catch(Exception e){
						// ignore if no parameters found
						System.out.println("No parameters loaded for this evaluation - network is not yet trained?");
					}
				}
			}
			
			// Create evaluation strategy
			strategy = factory.create(this.config.strategy);
			if(strategy == null)
				throw new Exception("Strategy "+this.config.strategy+" not available");
			
			strategy.setup(config, d, nns);
			
		
			tStart = System.currentTimeMillis();
			
			for(int i=0;i<d.size();i++){
				progress = strategy.processIteration(i);
				
				// TODO how frequently publish progress
				if(i % 1000 == 0){
					listenerExecutor.execute(()->{
						List<EvaluatorListener> copy = new ArrayList<>();
						synchronized(listeners){
							copy.addAll(listeners);
						}
						for(EvaluatorListener l : copy){
							l.onProgress(evaluatorId, progress);
						}
					});
				}
			}
			tEnd = System.currentTimeMillis();
			
			long evaluationTime = tEnd-tStart;
			
			Evaluation eval = strategy.getResult();
			eval.evaluationTime = evaluationTime;
			
			if(eval.error < this.config.storeIfBetterThan){
				for(NeuralNetwork nn : nns)
					nn.storeParameters(this.config.tag, "best");
			}
			
			return eval;
		} catch(Throwable t){
			System.err.println("Error during evaluation");
			List<EvaluatorListener> copy = new ArrayList<>();
			synchronized(listeners){
				copy.addAll(listeners);
			}
			for(EvaluatorListener l : copy){
				l.onException(evaluatorId, t);
			}
			throw t;
		}finally {
			List<EvaluatorListener> copy = new ArrayList<>();
			synchronized(listeners){
				copy.addAll(listeners);
			}
			EvaluationProgress progress =  getProgress();
			for(EvaluatorListener l : copy){
				l.onFinish(evaluatorId, progress);
			}
			evaluating = false;
			
			datasets.releaseDataset(d);
			System.gc();
		}
	}
	
	public EvaluationProgress getProgress(){
		if(!evaluating)
			return null;
		
		return progress;
	}
	
	public boolean isBusy(){
		return evaluating;
	}

	@Activate
	public void activate(BundleContext context){
		this.evaluatorId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
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
	void setEvaluationStrategyFactory(StrategyFactory f){
		factory = f;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addListener(EvaluatorListener listener, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			boolean listen = false;
			for(String target : targets){
				if(evaluatorId.toString().equals(target)){
					listen  = true;
				}
			}
			if(!listen)
				return;	
		}
		this.listeners.add(listener);
	}
	
	void removeListener(EvaluatorListener listener, Map<String, Object> properties){
		this.listeners.remove(listener);
	}

}

