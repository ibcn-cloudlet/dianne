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
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.Configuration;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
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
	
	private Map<String, Dataset> datasets = new HashMap<String, Dataset>();

	private Map<Dataset, List<Configuration>> adapters = new HashMap<Dataset, List<Configuration>>();
	
	private Map<String, DirectoryWatcher> watchers = new HashMap<>();
	private Map<String, Configuration> configs = new HashMap<>();
	
	@Activate
	void activate(BundleContext context) {
		String d = context.getProperty("be.iminds.iot.dianne.datasets.location");
		if(d != null){
			path = d;
		}
		
		if(path != null){
			final File dir = new File(path);
			if(!dir.exists())
				dir.mkdirs();
			Thread t = new Thread(() -> searchDatasetConfigs(dir, true));
			t.start();
		}
	}
	
	@Deactivate
	void deactivate(){
		watchers.values().forEach(watcher -> watcher.close());
	}
	
	@Override
	public List<DatasetDTO> getDatasets() {
		return datasets.values().stream().map(dataset -> dataset.getDTO()).collect(Collectors.toList());
	}

	@Override
	public Dataset getDataset(String name) {
		if(!datasets.containsKey(name))
			return null; // throw exception?
		
		return datasets.get(name);
	}
	
	@Override
	public Dataset configureDataset(String name, Map<String, String> config) {
		System.out.println("Dataset");
		System.out.println("---");
		System.out.println("* dataset = "+name);

		Configuration datasetConfiguration = null;
		List<Configuration> adapterConfigurations = new ArrayList<>();

		Dataset d = getDataset(name);
		if(d == null){
			// Try to create it ourselves
			Hashtable<String, Object> props = new Hashtable<>();
			
			// Mostly used to create experience pools...
			String pid = "be.iminds.iot.dianne.dataset.MemoryExperiencePool";
			if(config.containsKey("type")){
				String type = config.get("type");
				pid = "be.iminds.iot.dianne.dataset."+type;
			}

			String dir = config.get("dir");
			if(dir == null) {
				File dr = new File(path+"/"+name);
				dr.mkdirs();
				dir = dr.getAbsolutePath();
			}
			props.put("dir", dir);
			
			props.put("name", name);
			props.put("aiolos.combine", "*");
			props.put("aiolos.instance.id", name);

			for(Entry<String, String> e : config.entrySet()) {
				String key = e.getKey();
				String value = e.getValue();
				if(value.contains(",")) {
					props.put(key, value.split(",")); // treat comma separation as array type
				} else {
					props.put(key, value);
				}
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				datasetConfiguration = c;
			} catch(Exception e){
				e.printStackTrace();
			}
		} 
		
		// TODO type safe creation of dataset adapter configurations?
		// flip -> rotate -> crop -> frame?
		String adapter = name;
		// TODO range adapter
		
		if(config.containsKey("range")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.RangeAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			String s = config.get("range");
			System.out.println("* range = "+s);
			props.put("range", s.contains(",") ? s.split(",") : s);
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		if(config.containsKey("vflip") || config.containsKey("hflip")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.RandomFlipAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("vflip")){
				String s = config.get("vflip");
				props.put("vflip", s);
				System.out.println("* vflip = "+s);
			}

			if(config.containsKey("hflip")){
				String s = config.get("hflip");
				props.put("hflip", s);
				System.out.println("* hflip = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		} 
		if(config.containsKey("rotationTheta")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.RandomRotationAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("rotationTheta")){
				String s = config.get("rotationTheta");
				if(s.contains(",")){
					props.put("rotationTheta", s.split(","));
				} else {
					props.put("rotationTheta", s);
				}
				System.out.println("* rotationTheta = "+s);
			}
			if(config.containsKey("rotationCenter")){
				String s = config.get("rotationCenter");
				if(s.contains(",")){
					props.put("rotationCenter", s.split(","));
				} else {
					props.put("rotationCenter", s);
				}
				System.out.println("* rotationCenter = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		if(config.containsKey("cropWidth") || config.containsKey("cropHeight")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.RandomCropAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("cropWidth")){
				String s = config.get("cropWidth");
				if(s.contains(",")){
					props.put("cropWidth", s.split(","));
				} else {
					props.put("cropWidth", s);
				}
				System.out.println("* cropWidth = "+s);
			}
			if(config.containsKey("cropHeight")){
				String s = config.get("cropHeight");
				if(s.contains(",")){
					props.put("cropHeight", s.split(","));
				} else {
					props.put("cropHeight", s);
				}
				System.out.println("* cropHeight = "+s);
			}
			if(config.containsKey("cropPadding")){
				String s = config.get("cropPadding");
				props.put("cropPadding", s);
				System.out.println("* cropPadding = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		if(config.containsKey("frame")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.FrameAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("frame")){
				String s = config.get("frame");
				props.put("frame", s.split(","));
				System.out.println("* frame = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		if(config.containsKey("binarize")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.BinarizeAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("binarize")){
				String s = config.get("binarize");
				props.put("binarize", s);
				System.out.println("* binarize = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		if(config.containsKey("autoencode")){
			String pid = "be.iminds.iot.dianne.dataset.adapters.AutoencoderAdapter";
			Hashtable<String, Object> props = new Hashtable<>();
			props.put("Dataset.target", "(name="+adapter+")");
			adapter = name+"-"+UUID.randomUUID();
			props.put("name", adapter);
			props.put("aiolos.instance.id", adapter);
			props.put("aiolos.combine", "*");
			props.put("aiolos.export", "false");

			if(config.containsKey("autoencode")){
				String s = config.get("autoencode");
				props.put("autoencode", s);
				System.out.println("* autoencode = "+s);
			}
			if(config.containsKey("encoder_noise")){
				String s = config.get("encoder_noise");
				props.put("encoder_noise", s);
				System.out.println("* encoder_noise = "+s);
			}
			if(config.containsKey("encoder_drop")){
				String s = config.get("encoder_drop");
				props.put("encoder_drop", s);
				System.out.println("* encoder_drop = "+s);
			}
			
			try {
				Configuration c = ca.createFactoryConfiguration(pid, null);
				c.update(props);
				adapterConfigurations.add(c);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		// TODO delegate adater creation to separate factory services?
		// this would allow this code to reside in the rl bundle...
		if(config.containsKey("exploration")){
			String target = adapter;
			adapter = name+"-exploration="+config.get("exploration");
			if(getDataset(adapter) == null){
				String pid = "be.iminds.iot.dianne.dataset.adapters.HashcodeExplorationAdapter";
				Hashtable<String, Object> props = new Hashtable<>();
				props.put("Dataset.target", "(name="+target+")");
				props.put("name", adapter);
				props.put("aiolos.instance.id", adapter);
				props.put("aiolos.combine", "*");
				props.put("aiolos.export", "false");
	
				if(config.containsKey("exploration")){
					String s = config.get("exploration");
					props.put("exploration", s);
					System.out.println("* exploration = "+s);
				}
				
				try {
					Configuration c = ca.createFactoryConfiguration(pid, null);
					c.update(props);
					adapterConfigurations.add(c);
				} catch(Exception e){
					e.printStackTrace();
				}
			}
		}
		
		if(config.containsKey("stateSize") && config.containsKey("posterior")){
			String target = adapter;
			adapter = name+"-sb="+config.get("stateSize");
			if(getDataset(adapter) == null){
				String pid = "be.iminds.iot.dianne.dataset.adapters.StateBeliefAdapter";
				Hashtable<String, Object> props = new Hashtable<>();
				props.put("Dataset.target", "(name="+target+")");
				props.put("name", adapter);
				props.put("aiolos.instance.id", adapter);
				props.put("aiolos.combine", "*");
				props.put("aiolos.export", "false");
	
				if(config.containsKey("stateSize")){
					String s = config.get("stateSize");
					props.put("stateSize", s);
					System.out.println("* stateSize = "+s);
				}
				
				if(config.containsKey("sampleSize")){
					String s = config.get("sampleSize");
					props.put("sampleSize", s);
					System.out.println("* sampleSize = "+s);
				}
				
				if(config.containsKey("posterior")){
					String s = config.get("posterior");
					props.put("posterior", s);
					System.out.println("* posterior = "+s);
				} 
				
				if(config.containsKey("prior")){
					String s = config.get("prior");
					props.put("prior", s);
					System.out.println("* prior = "+s);
				}
				
				if(config.containsKey("tag")){
					String s = config.get("tag");
					props.put("tag", s);
					System.out.println("* tag = "+s);
				}
				
				try {
					Configuration c = ca.createFactoryConfiguration(pid, null);
					c.update(props);
					adapterConfigurations.add(c);
				} catch(Exception e){
					e.printStackTrace();
				}
			}
		}
		System.out.println("---");
		
		// now wait for the adapter dataset to come online
		d = getDataset(adapter);
		if(d == null){
			// TODO how long before timeout here?
			long t = System.currentTimeMillis();
			while(d == null && System.currentTimeMillis()-t < 10000){
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
				}
				d = getDataset(adapter);
			}
		}

		if(d != null){
			adapters.put(d, adapterConfigurations);
			// TODO should we cleanup the xp pool configuration after done or not?
		} else {
			// cleanup configurations if dataset didn't came online for some reason
			for(Configuration c : adapterConfigurations){
				try {
					c.delete();
				} catch (IOException e) {
				}
			}
			if(datasetConfiguration != null){
				try {
					datasetConfiguration.delete();
				} catch (IOException e) {
				}
			}
		}
		
		return d;
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
	

	@Override
	public boolean isClassificationDataset(String dataset) {
		Dataset d = datasets.get(dataset);
		if(d == null)
			return false;
		
		return d.getLabels() != null;
	}
	
	@Override
	public boolean isExperiencePool(String dataset){
		Dataset d = datasets.get(dataset);
		if(d == null)
			return false;
		
		// way to check whether it is an XP pool without having to import the interace
		try {
			d.getClass().getMethod("stateDims");
			return true;
		} catch(NoSuchMethodException e){
		}
		return false;
	}

	
	private void searchDatasetConfigs(File file, boolean recurse){
		if(file.isDirectory()){
			watchers.put(file.getAbsolutePath(), new DirectoryWatcher(file, 
				p->{
					searchDatasetConfigs(p.toFile(), false);
				}, null, 
				p->{
					// TODO problem : when saving file on Linux, 
					// it returns a temporary filename instead of the actual filename
					// so if you just change .json and save it, the old dataset remains...
					Configuration c = configs.get(p.toAbsolutePath().toString());
					if(c != null){
						try {
							c.delete();
						} catch (Exception e) {
						}
					}
					DirectoryWatcher w = watchers.get(p.toAbsolutePath().toString());
					if(w != null){
						w.close();
					}
				}
			));
			
			for(File f : file.listFiles()){
				if(f.getName().endsWith(".json")){
					Configuration c = parseDatasetConfiguration(f);
					if(c != null){
						configs.put(f.getAbsolutePath(), c);
					}
				}
				
				// go one level deep
				if(f.isDirectory()){
					searchDatasetConfigs(f, false);
				}
			}
		} else {
			if(file.getName().endsWith(".json")){
				try {
					Configuration c = parseDatasetConfiguration(file);
					if(c != null){
						configs.put(file.getAbsolutePath(), c);
					}
				} catch(Exception e){}
			}
		}
	}
	
	
	private Configuration parseDatasetConfiguration(File f){
		try {
			// parse any adapter configurations from JSON and apply config?
			JsonParser parser = new JsonParser();
			JsonObject json = parser.parse(new JsonReader(new FileReader(f))).getAsJsonObject();
			
			String name = json.get("name").getAsString();
			if(name == null)
				return null;  // should have a name
			
			
			Hashtable<String, Object> props = new Hashtable<>();

			String dir = f.getParentFile().getAbsolutePath();
			props.put("dir", dir);
			
			String pid = null;
			
			if(json.has("adapter")){
				String adapter = json.get("adapter").getAsString();
				pid = adapter.contains(".") ? adapter : "be.iminds.iot.dianne.dataset.adapters."+adapter;
				// in case of adapter, set Dataset target: the dataset it is adapting
				if(json.has("targetFilter")){
					String filter = json.get("targetFilter").getAsString();
					props.put("Dataset.target", filter);
				} else {
					String dataset = json.get("dataset").getAsString();
					props.put("Dataset.target", "(name="+dataset+")");
				}
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
				} else if(name.equalsIgnoreCase("ImageNetValidation")){
					pid = "be.iminds.iot.dianne.dataset.ImageNet.validation";
				} else if(name.equalsIgnoreCase("ImageNetTraining")){
					pid = "be.iminds.iot.dianne.dataset.ImageNet.training";
				} else {
					pid = "be.iminds.iot.dianne.dataset."+name;
				}
			}
			
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
						val[i] = a.get(i).getAsString().trim();
					}
					props.put(e.getKey(), val);
				} else {
					props.put(e.getKey(), e.getValue().getAsString().trim());
				}
			});
			config.update(props);
			return config;
		} catch(Exception e){
			System.err.println("Error parsing Dataset config file: "+f.getAbsolutePath());
			e.printStackTrace();
			return null;
		}
	}
	
	@Reference
	void setConfigurationAdmin(ConfigurationAdmin ca){
		this.ca = ca;
	}

	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		if(dataset == null){
			System.out.println(name+" dataset is inproperly configured, null service...");
			return;
		}
		
		if(properties.containsKey("aiolos.framework.uuid")){
			this.datasets.put(name, 
					(Dataset) Proxy.newProxyInstance(
							this.getClass().getClassLoader(), 
							dataset.getClass().getInterfaces(), 
							new RemoteDatasetProxy(dataset)));
		} else {
			this.datasets.put(name, dataset);
		}
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
	}
}
