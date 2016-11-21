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

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;

import org.apache.felix.service.command.Descriptor;
import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetDTO;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=datasets",
				  "osgi.command.function=forward",
				  "osgi.command.function=sample",
				  "osgi.command.function=sequence",
				  "osgi.command.function=sequences",
				  "osgi.command.function=dump"},
		immediate=true)
public class DianneDatasetCommands {

	private static Random rand = new Random(System.currentTimeMillis());
	
	BundleContext context;
	
	// Dianne components
	Dianne dianne;
	DiannePlatform platform;
	DianneDatasets datasets;
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Descriptor("List available datasets.")
	public void datasets(){
		List<DatasetDTO> ds = datasets.getDatasets();
		
		if(ds.size()==0){
			System.out.println("No datasets available");
			return;
		}
		
		System.out.println("Available datasets:");
		int i = 0;
		for(DatasetDTO dataset : ds){
			System.out.println("["+(i++)+"] "+dataset.name+"\t"+dataset.size+" samples");
		}
	}
	
	@Descriptor("Forward a dataset sample through a neural network instance.")
	public void forward(
			@Descriptor("dataset name to fetch a sample from")
			String dataset, 
			@Descriptor("uuid of the neural network instance")
			String nnId, 
			@Descriptor("index of dataset sample")
			int sample, 
			@Descriptor("(optional) tags to attach to the forward call ")
			String...tags){

		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		
		final int index = sample == -1 ? rand.nextInt(d.size()) : sample;
		
		NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nni==null){
			System.out.println("Neural network instance "+nnId+" not deployed");
			return;
		}
		
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		if(nn==null){
			System.out.println("Neural network instance "+nnId+" not available");
			return;
		}
		
		final String[] labels = nn.getOutputLabels();

		// get input and forward
		try {
			Tensor in = d.getSample(index).input;
			long t1 = System.currentTimeMillis();
			nn.forward(null, null, in, tags).then(
				p -> {
						long t2 = System.currentTimeMillis();

						Tensor out = p.getValue().tensor;
						int clazz = TensorOps.argmax(out);
						float max = TensorOps.max(out);
						String label = labels[clazz];
					
						System.out.println("Sample "+index+" (with tags "+Arrays.toString(tags)+") classified as: "+label+" (probability: "+max+")");
						System.out.println("Forward time: "+(t2-t1)+" ms");
						
						return null;
					}
				);
		
			
		} catch(Exception e ){
			e.printStackTrace();
		} 
	}
	
	@Descriptor("Forward a random dataset sample through a neural network instance.")
	public void forward(
			@Descriptor("dataset name to fetch a sample from")
			String dataset, 
			@Descriptor("uuid of the neural network instance")
			String nnId,
			@Descriptor("(optional) tags to attach to the forward call ")
			String...tags){
		forward(dataset, nnId, -1, tags);
	}

	@Descriptor("Forward a dataset sample through a neural network instance.")
	public void forward(
			@Descriptor("dataset name to fetch a sample from")
			String dataset, 
			@Descriptor("index of the neural network instance (from the list command output)")
			int index, 
			@Descriptor("index of dataset sample")
			int sample, 
			@Descriptor("(optional) tags to attach to the forward call ")
			String... tags){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network deployed with index "+index);
			return;
		}
		String id = nns.get(index).id.toString();
		
		forward(dataset, id, sample, tags);
	}
	
	@Descriptor("Forward a random dataset sample through a neural network instance.")
	public void forward(
			@Descriptor("dataset name to fetch a sample from")
			String dataset, 
			@Descriptor("index of the neural network instance (from the list command output)")
			int index, 
			@Descriptor("(optional) tags to attach to the forward call ")
			String... tags){
		forward(dataset, index, -1, tags);
	}

	@Descriptor("Print out a random sample of the dataset")
	public void sample(
			@Descriptor("dataset name to fetch a sample from")
			String dataset){ 
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		
		int index = rand.nextInt(d.size());
		System.out.println("["+index+"] "+d.getSample(index).toString());
	}

	@Descriptor("Print out a sample of the dataset")
	public void sample(
			@Descriptor("dataset name to fetch a sample from")
			String dataset,
			@Descriptor("index to print")
			int index){ 
		
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		System.out.println("["+index+"] "+d.getSample(index).toString());
	}
	
	@Descriptor("Print out samples of the dataset")
	public void sample(
			@Descriptor("dataset name to fetch a sample from")
			String dataset,
			@Descriptor("start index")
			int start,
			@Descriptor("end index")
			int end){ 
		
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		for(int index=start;index<end;index++)
			System.out.println("["+index+"] "+d.getSample(index).toString());
	}
	
	@Descriptor("Print out a sequence of the dataset")
	public void sequence(
			@Descriptor("dataset name to fetch a sample from")
			String dataset,
			@Descriptor("sequence to print")
			int sequence){ 
		
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		if(!(d instanceof SequenceDataset)){
			System.out.println("Dataset "+dataset+" is not a sequence dataset");
			return;
		}
		
		try {
			List<?> s = ((SequenceDataset<?,?>)d).getSequence(sequence);
			int i = 0;
			for(Object o : s){
				System.out.println("["+(i++)+"] "+o.toString());
			}
		} catch(Throwable t){
			t.printStackTrace();
		}
	}

	@Descriptor("Print out the number of sequences of the dataset")
	public void sequences(
			@Descriptor("dataset get the number of sequences from")
			String dataset){ 
		
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		if(!(d instanceof SequenceDataset)){
			System.out.println("Dataset "+dataset+" is not a sequence dataset");
			return;
		}
		
		System.out.println(((SequenceDataset<?, ?>)d).sequences());
	}
	
	@Descriptor("Dump the content of an experience pool")
	public void dump(
			@Descriptor("The experience pool to dump")
			String dataset){ 
		
		Dataset d = datasets.getDataset(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		if(!datasets.isExperiencePool(dataset)){
			System.out.println("Dataset "+dataset+" is not an experience pool");
			return;
		}
		
		
		try {
			d.getClass().getMethod("dump").invoke(d);
		} catch(Throwable t){
			t.printStackTrace();
		}
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
