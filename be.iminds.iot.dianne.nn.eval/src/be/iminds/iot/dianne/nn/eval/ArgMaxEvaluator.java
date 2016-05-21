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
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
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
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(property={"aiolos.unique=true"})
public class ArgMaxEvaluator implements Evaluator {
	
	protected UUID evaluatorId;
	
	protected DataLogger logger;
	
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected String tag = null;
	protected boolean trace = false;
	protected boolean includeOutputs = false;
	
	protected volatile boolean evaluating = false;
	
	protected int sample = 0;
	protected int faulty = 0;
	protected int total = 0;
	protected Tensor confusion;
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
		
		try {
			// Fetch the dataset
			Dataset d = datasets.get(dataset);
			if(d==null){
				throw new Exception("Dataset "+dataset+" not available");
			}
			
			if(config.containsKey("tag")){
				tag = config.get("tag"); 
			}
			
			if(config.containsKey("trace")){
				trace = Boolean.parseBoolean(config.get("trace"));
			}
			
			if(config.containsKey("includeOutputs")){
				includeOutputs = Boolean.parseBoolean(config.get("includeOutputs"));
			}
			
			System.out.println("Evaluator Configuration");
			System.out.println("=======================");
			System.out.println("* dataset = "+dataset);
			System.out.println("* tag = "+tag);
			System.out.println("* trace = "+trace);
			System.out.println("* includeOutputs = "+includeOutputs);
			
			
			int[] indices = null;
			String range = config.get("range");
			if(range!=null){
				indices = parseRange(range);
				
				System.out.println("Dataset range");
				if(range.contains(":"))
					System.out.println("* range = "+range);
				else 
					System.out.println("* "+indices.length+" indices selected");
				System.out.println("---");
			} else {
				int startIndex = 0;
				int endIndex = d.size();
				
				String start = config.get("startIndex");
				if(start!=null){
					startIndex = Integer.parseInt(start);
				}
				
				String end = config.get("endIndex");
				if(end!=null){
					endIndex = Integer.parseInt(end);
				}
				
				int index = startIndex;
				indices = new int[endIndex-startIndex];
				for(int i=0;i<indices.length;i++){
					indices[i] = index++;
				}
				
				System.out.println("Dataset range");
				System.out.println("* startIndex = "+startIndex);
				System.out.println("* endIndex = "+endIndex);
				System.out.println("---");
			}
			
			total = indices.length;
			faulty = 0;
			
			NeuralNetwork nn = null;
			try {
				nn = dianne.getNeuralNetwork(nni).getValue();
			} catch (Exception e) {
				throw new Exception("Neural Network "+nni.id+" not available!");
			}
			nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
			
			try {
				if(tag==null){
					nn.loadParameters();
				} else {
					nn.loadParameters(tag);
				}
			} catch(Exception e){
				// ignore if no parameters found
				System.out.println("No parameters loaded for this evaluation - network is not yet trained?");
			}
		
			confusion = null;
			outputs = includeOutputs ? new ArrayList<Tensor>() : null;
			tStart = System.currentTimeMillis(); tForward = 0;
			
			Tensor in = null;
			for(sample=0;sample<indices.length;sample++){
				in = d.getInputSample(indices[sample], in);
				
				long t = System.nanoTime();
				Tensor out = nn.forward(in);
				tForward += System.nanoTime() - t;
				
				if(outputs!=null)
					outputs.add(out);
				
				if(confusion==null){
					int outputSize = out.size();
					confusion = new Tensor(outputSize, outputSize);
					confusion.fill(0.0f);
				}
				
				int predicted = TensorOps.argmax(out);
				int real = TensorOps.argmax(d.getOutputSample(indices[sample]));
				if(real!=predicted)
					faulty++;
				
				if(trace){
					System.out.println("Sample "+indices[sample]+" was "+predicted+", should be "+real);
				}
				
				confusion.set(confusion.get(real, predicted)+1, real, predicted);
			}
			tEnd = System.currentTimeMillis();
			
			Evaluation e = new Evaluation(total, faulty/(float)total, confusion, outputs, tEnd-tStart, (tForward/1000000f)/total);
			return e;
		} finally {
			evaluating = false;
		}
	}
	
	public EvaluationProgress getProgress(){
		if(!evaluating)
			return null;
		
		EvaluationProgress progress = new EvaluationProgress(sample, total, System.currentTimeMillis()-tStart, (tForward/1000000f)/total);
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
	
	private int[] parseRange(String range){
		ArrayList<Integer> list = new ArrayList<>();
		String[] subranges = range.split(",");
		for(String subrange : subranges){
			String[] s = subrange.split(":");
			if(s.length==2){
				for(int i=Integer.parseInt(s[0]);i<Integer.parseInt(s[1]);i++){
					list.add(i);
				}
			} else {
				list.add(Integer.parseInt(s[0]));
			}
		}
		int[] array = new int[list.size()];
		for(int i=0;i<list.size();i++){
			array[i] = list.get(i);
		}
		return array;
	}

}

