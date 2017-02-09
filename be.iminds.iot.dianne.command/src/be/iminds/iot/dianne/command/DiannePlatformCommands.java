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
import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;

import org.apache.felix.service.command.Descriptor;
import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=runtimes",
				  "osgi.command.function=list",
				  "osgi.command.function=tags",
				  "osgi.command.function=info",
				  "osgi.command.function=models",
				  "osgi.command.function=deploy",
				  "osgi.command.function=undeploy",
				  "osgi.command.function=size",
				  "osgi.command.function=gc"},
		immediate=true)
public class DiannePlatformCommands {

	BundleContext context;
	
	// Dianne components
	Dianne dianne;
	DiannePlatform platform;
	DianneRepository repository;
	
	// State
	Map<UUID, ServiceRegistration<RepositoryListener>> repoListeners = new HashMap<>();
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Descriptor("Trigger JVM garbage collection.")
	public void gc(){
		System.gc();
	}

	@Descriptor("List available runtimes to deploy modules on.")
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
	
	@Descriptor("List the available models in the repository.")
	public void models(){
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
	
	@Descriptor("List all deployed neural networks.")
	public void list(){
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

	@Descriptor("List all available tags for a given neural network.")
	public void tags(String nnName){
		Collection<String> tags = repository.listTags(nnName);
		for(String tag : tags){
			System.out.println("* "+tag);
		}
	}
	
	@Descriptor("Print details of a neural network instance.")
	public void info(
			@Descriptor("index of the neural network instance (from the list command output)")
			int index){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network deployed with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		printNN(nn);
	}
	
	@Descriptor("Print details of a neural network instance.")
	public void info(
			@Descriptor("uuid of the neural network instance")
			String id){
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(id));
		if(nn==null){
			System.out.println("No neural network deployed with id "+id);
			return;
		}
		printNN(nn);
	}
	
	private void printNN(NeuralNetworkInstanceDTO nn){
		System.out.println(nn.name+" ("+nn.id.toString()+")");
		for(ModuleInstanceDTO m: nn.modules.values()){
			System.out.println("* "+m.module.type+" ("+m.moduleId+") "+" deployed at "+m.runtimeId);
		}
	}
	
	@Descriptor("Deploy a neural network on the default runtime.")
	public void deploy(
			@Descriptor("name of the neural network")
			String name){
		deploy(name, platform.getRuntimes().keySet().iterator().next());
	}
	
	@Descriptor("Deploy a neural network on a runtime.")
	public void deploy(
			@Descriptor("name of the neural network")
			String name, 
			@Descriptor("uuid of the target runtime")
			String id){
		deploy(name, UUID.fromString(id));
	}
	
	@Descriptor("Deploy a neural network on a runtime.")
	public void deploy(
			@Descriptor("name of the neural network")
			String name, 
			@Descriptor("index of the target runtime (from the runtimes command output)")
			int index){
		List<UUID> runtimes = new ArrayList<UUID>(platform.getRuntimes().keySet());
		deploy(name, runtimes.get(index));
	}
	
	@Descriptor("Deploy a neural network on a runtime and load weights with specific tag.")
	public void deploy(
			@Descriptor("name of the neural network")
			String name, 
			@Descriptor("uuid of the target runtime")
			String id, 
			@Descriptor("tag of the weights to load")
			String tag){
		NeuralNetworkInstanceDTO nn = deploy(name, UUID.fromString(id));
		
		// load parameters with tag
		loadParameters(nn, tag);
		
		// add updatelistener for tag
		addRepositoryListener(nn, tag);
	}
	
	@Descriptor("Deploy a neural network on a runtime and load weights with specific tag.")
	public void deploy(
			@Descriptor("name of the neural network")
			String name, 
			@Descriptor("index of the target runtime (from the runtimes command output)")
			int index, 
			@Descriptor("tag of the weights to load")
			String tag){
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
	
	@Descriptor("Undeploy a neural network.")
	public void undeploy(
			@Descriptor("uuid of the neural network instance")
			String nnId){
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("No neural network deployed with id "+nnId);
			return;
		}
		undeploy(nn);
	}
	
	@Descriptor("Undeploy a neural network.")
	public void undeploy(
			@Descriptor("index of the neural network instance (from the list command output)")
			int index){
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
		
		ServiceRegistration<RepositoryListener> r = repoListeners.get(nn.id);
		if(r!=null){
			r.unregister();
		}
	}
	
	@Descriptor("Print size of a neural network.")
	public void size(
			@Descriptor("uuid of the neural network instance")
			String nnId){
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("No neural network deployed with id "+nnId);
			return;
		}
		size(nn);
	}
	
	@Descriptor("Print size of a neural network.")
	public void size(
			@Descriptor("index of the neural network instance (from the list command output)")
			int index){
		List<NeuralNetworkInstanceDTO> nns = platform.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		size(nn);
	}
	
	private void size(NeuralNetworkInstanceDTO nni){
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		if(nn!=null){
			Map<UUID, Tensor> params = nn.getParameters();
			int size = params.values().stream().mapToInt(t -> t.size()).sum();
			System.out.println("Size: "+size);
		}
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
		ServiceRegistration<RepositoryListener> r = context.registerService(RepositoryListener.class, listener, props);
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
	
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}

	@Reference
	void setDianneRepository(DianneRepository r){
		repository = r;
	}
	
}
