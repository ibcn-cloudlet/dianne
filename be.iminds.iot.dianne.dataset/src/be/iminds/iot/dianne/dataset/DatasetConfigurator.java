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
package be.iminds.iot.dianne.dataset;


import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.Configuration;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetDTO;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;


@Component(
	property={"aiolos.export=false"})
public class DatasetConfigurator implements DianneDatasets {

	private ConfigurationAdmin ca;
	
	private String path = "datasets";
	
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	protected Map<String, DatasetDTO> dtos = new HashMap<String, DatasetDTO>(); 

	protected Map<Dataset, List<Configuration>> adapters = new HashMap<Dataset, List<Configuration>>();
	
	@Activate
	void activate(BundleContext context) {
		String d = context.getProperty("be.iminds.iot.dianne.datasets.location");
		if(d != null){
			path = d;
		}
		
		File dir = new File(path);
		searchDirectory(dir, true);
	}
	
	@Override
	public List<DatasetDTO> getDatasets() {
		return dtos.values().stream().collect(Collectors.toList());
	}

	@Override
	public Dataset getDataset(String name) {
		if(!datasets.containsKey(name))
			return null; // throw exception?
		
		return datasets.get(name);
	}
	
	@Override
	public Dataset configureDataset(String name, Map<String, String> config) {
		Dataset d = getDataset(name);
		if(d == null)
			return null;
		
		// TODO apply adapter configurations and return adapter!
		// flip -> rotate -> crop -> frame?
		
		
		return datasets.get(name);
	}

	@Override
	public void releaseDataset(Dataset d) {
		// remove adapters
		List<Configuration> configurations = adapters.remove(d);
		if(configurations != null){
			for(Configuration c : configurations){
				try {
					c.delete();
				} catch (IOException e) {
				}
			}
		}
	}
	
	private void searchDirectory(File dir, boolean recurse){
		if(!dir.isDirectory())
			return;
		
		for(File f : dir.listFiles()){
			if(f.getName().endsWith(".json")){
				parseDatasetConfiguration(f);
			}
			
			// go one level deep
			if(f.isDirectory()){
				searchDirectory(f, false);
			}
		}
	}
	
	
	private void parseDatasetConfiguration(File f){
		try {
			// parse any adapter configurations from JSON and apply config?
			JsonParser parser = new JsonParser();
			JsonObject json = parser.parse(new JsonReader(new FileReader(f))).getAsJsonObject();
			
			String name = json.get("name").getAsString();
			if(name == null)
				return;  // should have a name
			
			
			Hashtable<String, Object> props = new Hashtable<>();

			String dir = f.getParentFile().getAbsolutePath();
			props.put("dir", dir);
			
			String pid = null;
			
			if(json.has("adapter")){
				String adapter = json.get("adapter").getAsString();
				pid = "be.iminds.iot.dianne.dataset.adapters."+adapter;
				// in case of adapter, set Dataset target: the dataset it is adapting
				String dataset = json.get("dataset").getAsString();
				props.put("Dataset.target", "(name="+dataset+")");
			} else if(json.has("type")){
				String type = json.get("type").getAsString();
				pid = "be.iminds.iot.dianne.dataset."+type;
			} else {
				// some hard coded pids
				if(name.startsWith("MNIST")){
					pid = "be.iminds.iot.dianne.dataset.MNIST";
				} else if(name.startsWith("CIFAR-100")){
					pid = "be.iminds.iot.dianne.dataset.CIFAR100";
				} else if(name.startsWith("CIFAR-10")){
					pid = "be.iminds.iot.dianne.dataset.CIFAR10";
				} else if(name.startsWith("STL-10")){
					pid = "be.iminds.iot.dianne.dataset.STL10";
				} else if(name.startsWith("SVHN")){
					pid = "be.iminds.iot.dianne.dataset.SVHN";
				} else if(name.startsWith("ImageNet")){
					pid = "be.iminds.iot.dianne.dataset.ImageNet";
				}
			}
			
			if(pid == null)
				return;
			
			// set an aiolos instance id using the dataset name to treat
			// equally named datasets as single instance in the network
			props.put("aiolos.instance.id", name);
			// combine all offered interfaces (might be SequenceDataset or ExperiencePool)
			props.put("aiolos.combine", "*");
				
			// TODO use object conversion from JSON here?
			Configuration config = ca.createFactoryConfiguration(pid, null);
			json.entrySet().stream().forEach(e -> {
				if(e.getValue().isJsonArray()){
					JsonArray a = e.getValue().getAsJsonArray();
					String[] val = new String[a.size()];
					for(int i=0;i<val.length;i++){
						val[i] = a.get(i).getAsString();
					}
					props.put(e.getKey(), val);
				} else {
					props.put(e.getKey(), e.getValue().getAsString());
				}
			});
			config.update(props);
		} catch(Exception e){
			System.err.println("Error parsing Dataset config file: "+f.getAbsolutePath());
			e.printStackTrace();
		}
	}
	
	@Reference
	void setConfigurationAdmin(ConfigurationAdmin ca){
		this.ca = ca;
	}

	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		DatasetDTO dto = new DatasetDTO();
		dto.name = dataset.getName();
		dto.inputDims = dataset.inputDims();
		dto.targetDims = dataset.targetDims();
		dto.size = dataset.size();
		dto.labels = dataset.getLabels();
		
		this.datasets.put(dto.name, dataset);
		this.dtos.put(dto.name, dto);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
		this.dtos.remove(name);
	}
}
