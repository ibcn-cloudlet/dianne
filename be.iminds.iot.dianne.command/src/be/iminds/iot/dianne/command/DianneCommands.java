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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.Map.Entry;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=datasets",
				  "osgi.command.function=runtimes",
				  "osgi.command.function=nn",
				  "osgi.command.function=nnAvailable",
				  "osgi.command.function=nnDeploy",
				  "osgi.command.function=nnUndeploy",
				  "osgi.command.function=sample"},
		immediate=true)
public class DianneCommands {

	private static Random rand = new Random(System.currentTimeMillis());
	
	BundleContext context;
	
	// Dianne components
	TensorFactory factory; 
	
	Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());
	Dianne dianne;
	DiannePlatform platform;
	
	// State
	Map<UUID, ServiceRegistration> repoListeners = new HashMap<UUID, ServiceRegistration>();
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	public void datasets(){
		if(datasets.size()==0){
			System.out.println("No datasets available");
			return;
		}
		
		System.out.println("Available datasets:");
		synchronized(datasets){
			int i = 0;
			for(Dataset dataset : datasets.values()){
				System.out.println("["+(i++)+"] "+dataset.getName()+"\t"+dataset.size()+" samples");
			}
		}
	}
	
	public void runtimes(){
		if(platform.getRuntimes().size()==0){
			System.out.println("No runtimes available");
			return;
		}
		
		System.out.println("Available Dianne runtimes:");
		int i = 0;
		
		for(Entry<UUID, String> runtime : platform.getRuntimes().entrySet()){
			System.out.println("["+(i++)+"] "+runtime.getKey()+" - "+runtime.getValue());
		}
	}
	
	public void nnAvailable(){
		List<String> nns = platform.getAvailableNeuralNetworks();
		if(nns.size()==0){
			System.out.println("No neural networks available");
			return;
		}
		
		System.out.println("Available neural networks:");
		int i=0;
		for(String nn : nns){
			System.out.println("["+(i++)+"] "+nn);
		}
	}
	
	public void nn(){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(nns.size()==0){
			System.out.println("No neural networks deployed");
			return;
		}
		
		System.out.println("Deployed neural networks:");
		int i=0;
		for(NeuralNetworkInstanceDTO nn : nns){
			System.out.println("["+(i++)+"] "+nn.id+"\t"+nn.name);
		}
		
	}
	
	public void nn(int index){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network deployed with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		printNN(nn);
	}
	
	public void nn(String id){
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(id));
		if(nn==null){
			System.out.println("No neural network deployed with id "+id);
			return;
		}
		printNN(nn);
	}
	
	private void printNN(NeuralNetworkInstanceDTO nn){
		System.out.println(nn.id.toString()+" ("+nn.name+")");
		for(ModuleInstanceDTO m: nn.modules.values()){
			System.out.println("* "+m.moduleId+" deployed at "+m.runtimeId);
		}
	}
	
	public void nnDeploy(String name){
		deploy(name, platform.getRuntimes().keySet().iterator().next());
	}
	
	public void nnDeploy(String name, String id){
		deploy(name, UUID.fromString(id));
	}
	
	public void nnDeploy(String name, int index){
		List<UUID> runtimes = new ArrayList<UUID>(platform.getRuntimes().keySet());
		deploy(name, runtimes.get(index));
	}
	
	public void nnDeploy(String name, String id, String tag){
		NeuralNetworkInstanceDTO nn = deploy(name, UUID.fromString(id));
		
		// load parameters with tag
		loadParameters(nn, tag);
		
		// add updatelistener for tag
		addRepositoryListener(nn, tag);
	}
	
	public void nnDeploy(String name, int index, String tag){
		List<UUID> runtimes = new ArrayList<UUID>(platform.getRuntimes().keySet());
		NeuralNetworkInstanceDTO nn = deploy(name, runtimes.get(index));
		
		// load parameters with tag
		loadParameters(nn, tag);
		
		// add updatelistener for tag
		addRepositoryListener(nn, tag);
	}
	

	private synchronized NeuralNetworkInstanceDTO deploy(String name, UUID runtimeId){
		try {
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(name, runtimeId);
			System.out.println("Deployed instance of "+nni.name+" ("+nni.id.toString()+")");
			return nni;
		} catch (InstantiationException e) {
			System.out.println("Error deploying instance of "+name);
			e.printStackTrace();
		}
		return null;
	}
	
	public void nnUndeploy(String nnId){
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("No neural network deployed with id "+nnId);
			return;
		}
		undeploy(nn);
	}
	
	public void nnUndeploy(int index){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		undeploy(nn);
	}
	
	private void undeploy(NeuralNetworkInstanceDTO nn){
		platform.undeployNeuralNetwork(nn);
		
		ServiceRegistration r = repoListeners.get(nn.id);
		if(r!=null){
			r.unregister();
		}
	}
	
	public void sample(String dataset, String nnId, int sample, String...tags){

		Dataset d = datasets.get(dataset);
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
			Tensor in = d.getInputSample(index);
			long t1 = System.currentTimeMillis();
			nn.forward(null, null, in, tags).then(
				p -> {
						long t2 = System.currentTimeMillis();

						Tensor out = p.getValue().tensor;
						int clazz = factory.getTensorMath().argmax(out);
						float max = factory.getTensorMath().max(out);
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
	
	public void sample(String dataset, String nnId, String...tags){
		sample(dataset, nnId, -1, tags);
	}

	private void loadParameters(NeuralNetworkInstanceDTO nni, String tag){
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		if(nn!=null){
			try {
				nn.loadParameters(tag);
			} catch(Exception e){
				System.out.println("Failed to load parameters with tag "+tag);
			}
		}
	}
	
	private void addRepositoryListener(NeuralNetworkInstanceDTO nni, String tag){
		ParameterUpdateListener listener = new ParameterUpdateListener(nni);
		Dictionary<String, Object> props = new Hashtable<String, Object>();
		props.put("targets", new String[]{":"+tag});
		props.put("aiolos.unique", true);
		ServiceRegistration r = context.registerService(RepositoryListener.class, listener, props);
		repoListeners.put(nni.id, r);
	}
	
	class ParameterUpdateListener implements RepositoryListener {

		final NeuralNetworkInstanceDTO nni;
		
		public ParameterUpdateListener(NeuralNetworkInstanceDTO nni) {
			this.nni = nni;
		}
		
		@Override
		public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds,
				String... tag) {
			NeuralNetwork nn = null;
			try {
				nn = dianne.getNeuralNetwork(nni).getValue();
			} catch (Exception e) {
			}
			if(nn!=null){
				try {
					nn.loadParameters(tag);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
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
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}

	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
