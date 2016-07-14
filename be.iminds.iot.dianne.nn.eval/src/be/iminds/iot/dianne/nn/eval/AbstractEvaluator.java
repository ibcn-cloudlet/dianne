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
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.eval.EvaluatorListener;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.nn.eval.config.EvaluatorConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The AbstractEvaluator has all mechanics to loop through (part of) a Dataset
 * and forward each sample.
 * 
 * Concrete evaluators should implement the evalOutput(index, output, expected output)
 * and update the error/confusionMatrix
 * @author tverbele
 *
 */
public abstract class AbstractEvaluator implements Evaluator {
	
	private static ExecutorService listenerExecutor = Executors.newSingleThreadExecutor(); 
	protected List<EvaluatorListener> listeners = Collections.synchronizedList(new ArrayList<>());
	
	protected UUID evaluatorId;
	
	protected DataLogger logger;
	
	protected Dianne dianne;
	protected DianneDatasets datasets;
	
	protected EvaluatorConfig config;
	
	protected volatile boolean evaluating = false;
	
	protected int index = 0;
	protected int total;
	protected float error;
	protected List<Tensor> outputs;
	protected long tStart, tEnd, tForward;
	
	@Override
	public UUID getEvaluatorId(){
		return evaluatorId;
	}

	@Override
	public synchronized Evaluation eval(String dataset,
			Map<String, String> config, NeuralNetworkInstanceDTO nni) throws Exception {
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
			if(d==null){
				throw new Exception("Dataset "+dataset+" not available");
			}
			total = d.size();
			error = 0.0f;
			init(config);
			
			NeuralNetwork nn = null;
			try {
				nn = dianne.getNeuralNetwork(nni).getValue();
			} catch (Exception e) {
				throw new Exception("Neural Network "+nni.id+" not available!");
			}
			
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
		
			outputs = this.config.includeOutputs ? new ArrayList<Tensor>() : null;
			tStart = System.currentTimeMillis(); tForward = 0;
			
			Sample sample = null;
			Batch batch = null;
			int[] indices = null;
			boolean batchMode = false;
			if(this.config.batchSize > 0 && d.inputDims() != null){
				batchMode = true;
				indices = new int[this.config.batchSize];
			}
			
			for(index=0;index<total;index++){
				Tensor out;
				if(batchMode){
					// for last batch, adjust to remaining size
					if(index+indices.length > total){
						indices = new int[total-index];
						batch = null;
					}
					for(int i=0;i<indices.length;i++){
						indices[i] = index++;
					}
					index--; // dont count one too much
					batch = d.getBatch(batch, indices);
					
					long t = System.nanoTime();
					out = nn.forward(batch.input);
					tForward += System.nanoTime() - t;
					
					if(outputs!=null){
						for(int i=0;i<this.config.batchSize;i++)
							outputs.add(out.select(0, i).copyInto(null));
					}
					
					float err = evalOutput(out, batch.target);
					if(this.config.trace){
						System.out.println("Batch "+index/this.config.batchSize+" has avg error "+err/this.config.batchSize);
					}
					error += err;

				} else {
					sample = d.getSample(sample, index);
					
					long t = System.nanoTime();
					out = nn.forward(sample.input);
					tForward += System.nanoTime() - t;
					
					if(outputs!=null)
						outputs.add(out.copyInto(null));

					float err = evalOutput(out, sample.target);
					error += err;

					if(this.config.trace){
						System.out.println("Sample "+index+" has error "+err);
					}
				}
				
				
				if(index % 1000 == 0){
					listenerExecutor.execute(()->{
						List<EvaluatorListener> copy = new ArrayList<>();
						synchronized(listeners){
							copy.addAll(listeners);
						}
						EvaluationProgress progress =  getProgress();
						for(EvaluatorListener l : copy){
							l.onProgress(evaluatorId, progress);
						}
					});
				}
			}
			tEnd = System.currentTimeMillis();
			
			long evaluationTime = tEnd-tStart;
			float forwardTime =  (tForward/1000000f)/total;
			
			Evaluation eval = finish();
			eval.total = total;
			eval.error = error/total;
			eval.forwardTime = forwardTime;
			eval.evaluationTime = evaluationTime;
			eval.outputs = outputs;
			
			if(eval.error < this.config.storeIfBetterThan){
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
	
	protected abstract void init(Map<String, String> config);
	
	protected abstract float evalOutput(Tensor out, Tensor expected);
	
	protected abstract Evaluation finish();
	
	public EvaluationProgress getProgress(){
		if(!evaluating)
			return null;
		
		EvaluationProgress progress = new EvaluationProgress(index, total, error/index, System.currentTimeMillis()-tStart, (tForward/1000000f)/total);
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
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	void setDataLogger(DataLogger l){
		this.logger = l;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	protected void addListener(EvaluatorListener listener, Map<String, Object> properties){
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
	
	protected void removeListener(EvaluatorListener listener, Map<String, Object> properties){
		this.listeners.remove(listener);
	}

}

