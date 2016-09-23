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
package be.iminds.iot.dianne.command;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.felix.service.command.Descriptor;
import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=benchmark",
				  "osgi.command.function=trace"},
		immediate=true)
public class DianneBenchmarkCommands {

	private static DecimalFormat df = new DecimalFormat("0.###");
	
	Random random = new Random(System.currentTimeMillis());
	
	BundleContext context;
	
	// Dianne components
	Dianne dianne;
	DiannePlatform platform;
	DianneDatasets datasets;
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	
	@Descriptor("Benchmark a neural network.")
	public void benchmark(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input,
			@Descriptor("number of runs to execute")
			int runs,
			@Descriptor("times to forward in one run")
			int times,
			@Descriptor("runs to warmup")
			int warmup,
			@Descriptor("also include a backward pass")
			boolean backward
		){

		// parse the input
		Tensor in = null;
		
		String datasetName = input;
		String sample = null;
		if(input.contains(":")){
			int split = input.lastIndexOf(":");
			datasetName = input.substring(0, split);
			sample = input.substring(split + 1);
		}
		
		Dataset dataset = datasets.getDataset(datasetName);
		if(dataset != null){
			if(sample == null){
				in = dataset.getSample(random.nextInt(dataset.size())).input;
			} else {
				if( sample.contains(",")){
					String[] is = sample.split(",");
					int start = Integer.parseInt(is[0]);
					int count = Integer.parseInt(is[1]);
					int[] indices = new int[count];
					for(int i=0;i<count;i++){
						indices[i] = start++;
					}
					in = dataset.getBatch(indices).input;
				} else {
					in = dataset.getSample(Integer.parseInt(sample)).input;
				}
			}
		} else {
			// input dims are given
			int[] dims = null;
			try {
				String[] d = input.split(",");
				dims = new int[d.length];
				for(int i=0;i<d.length;i++){
					dims[i] = Integer.parseInt(d[i]);
				}
			} catch(Exception e){
				System.out.println("Incorrect dimensions provided...");
				return;
			}
			
			in = new Tensor(dims);
			in.rand();
		}
		
		if(in == null){
			System.out.println("No valid input provided...");
			return;
		}
		
		// deploy the NN
		NeuralNetworkInstanceDTO nni = null;
		try {
			nni = platform.deployNeuralNetwork(nnName);
		} catch (InstantiationException e1) {
			System.out.println("Neural network "+nnName+" could not be deployed...");
			return;
		}
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
			platform.undeployNeuralNetwork(nni.id);
		}
		if(nn==null){
			System.out.println("Neural network "+nnName+" could not be deployed...");
			return;
		}
		
		List<Double> timings = new ArrayList<>();
		try {
			for(int i=0;i<warmup;i++){
				run(nn, in, times, backward);
			}
			
			for(int i=0;i<runs;i++){
				timings.add(run(nn, in, times, backward));
			}
		
		} catch(Exception e){
			System.out.println("Error running the benchmark: "+e.getMessage());
			e.printStackTrace();
		} finally {
			platform.undeployNeuralNetwork(nni.id);
		}
		
		System.out.println("Benchmark "+nnName+" ("+times+" times - "+runs+" runs):");
		if(runs > 1){
			double avg = timings.stream().mapToDouble(t -> t).sum()/timings.size();
			double s = timings.stream().mapToDouble(t -> t).map(t -> (t-avg)).map(t->t*t).sum();
			
			System.out.println("Average run time: "+df.format(avg)+" ms");
			System.out.println("Standard deviation: "+df.format(Math.sqrt(s/(timings.size()-1)))+" ms");
		} else {
			System.out.println("Run time: "+df.format(timings.get(0))+" ms");
		}
	}
	
	@Descriptor("Benchmark a neural network.")
	public void benchmark(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input){
		benchmark(nnName, input, 30, 1, 10, false);
	}

	@Descriptor("Benchmark a neural network.")
	public void benchmark(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input,
			@Descriptor("number of runs to execute")
			int runs){
		benchmark(nnName, input, runs, 1, 10, false);
	}

	@Descriptor("Benchmark a neural network.")
	public void benchmark(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input,
			@Descriptor("number of runs to execute")
			int runs,
			@Descriptor("times to forward in one run")
			int times){
		benchmark(nnName, input, runs, times, 10, false);
	}
	
	@Descriptor("Set module tracing on/off.")
	public void trace(
			@Descriptor("set trace on/off")
			boolean on){
		AbstractModule.TRACE = on;
	}
	
	@Descriptor("Trace a neural network.")
	public void trace(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input,
			@Descriptor("also include a backward pass")
			boolean backward
			){
		boolean trace = AbstractModule.TRACE;
		AbstractModule.TRACE = true;
		
		benchmark(nnName, input, 1, 1, 0, backward);
		
		AbstractModule.TRACE = trace;
	}
	

	@Descriptor("Trace a neural network.")
	public void trace(
			@Descriptor("neural network to benchmark")
			String nnName, 
			@Descriptor("neural network input, either dims (comma separated e.g. 10,28,28) or dataset (datasetName[:sample,batchSize])")
			String input
			){
		trace(nnName, input, false);
	}
	
	private double run(NeuralNetwork nn, Tensor input, int times, boolean backward) throws Exception {
		long t1 = System.nanoTime();
		Tensor result = null;
		for(int i=0;i<times;i++)
			result = nn.forward(null, null, input).then(
				p -> {
					if(backward){
						Tensor out = p.getValue().tensor;
						return nn.backward(null, null, out);
					} else {
						return p;
					}
				}).then(
				p -> {	
					// acc grad
					if(backward)
						nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
					return p;
				}).getValue().tensor;
		long t2 = System.nanoTime();
		if(result == null){
			throw new Exception("Null result?!");
		}
		return (t2-t1)/1e6;
	}
	
	@Reference
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}

}
