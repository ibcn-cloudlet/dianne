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
package be.iminds.iot.dianne.repository.file;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(immediate=true)
public class DianneFileRepository implements DianneRepository {

	private String dir = "models";
	
	private Map<RepositoryListener, List<String>> listeners = Collections.synchronizedMap(new HashMap<RepositoryListener, List<String>>());
	protected ExecutorService executor = Executors.newSingleThreadExecutor();
	
	private final DianneRepositoryLock lock = new DianneRepositoryLock();
	
	@Activate
	public void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.storage");
		if(s!=null){
			dir = s;
		}
		File d = new File(dir+"/weights/");
		d.mkdirs();
	}
	
	@Deactivate
	public void deactivate(){
		executor.shutdownNow();
	}
	
	@Override
	public List<String> availableNeuralNetworks() {
		List<String> nns = new ArrayList<String>();
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(!f.isDirectory()){
				try (ZipFile zip = new ZipFile(f)){
					if(zip.getEntry("modules.txt")==null)
						continue;
				} catch(IOException e){
					continue;
				}
			}
			
			String name = f.getName();
			if(name.contains(".")){
				name = name.substring(0, name.indexOf('.'));
			}
			
			if(!name.equals("weights")){
				nns.add(name);
			}
		}
		return nns;
	}

	@Override
	public NeuralNetworkDTO loadNeuralNetwork(String nnName){
		String nn = null;
		File nnDir = new File(dir+"/"+nnName);
		if(nnDir.exists()){
			try {
				nn = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/modules.txt")));
			} catch (IOException e) {
				throw new RuntimeException("Failed to load neural network "+nnName, e);
			}
		} else {
			String path = findZip(nnName);
			nn = new String(findInZip(path, "modules.txt"));
		}
		return DianneJSONConverter.parseJSON(nn);
	}
	
	@Override
	public void storeNeuralNetwork(NeuralNetworkDTO nn){
		File d = new File(dir+"/"+nn.name);
		d.mkdirs();
		
		File n = new File(dir+"/"+nn.name+"/modules.txt");
		
		File locked = new File(dir+"/"+nn.name+"/.locked");
		if(locked.exists()){
			throw new RuntimeException("Cannot store neural network "+nn.name+", this is locked!");
		}
		
		String output = DianneJSONConverter.toJsonString(nn, true);
		
		try(PrintWriter p = new PrintWriter(n)) {
			p.write(output);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public String loadLayout(String nnName) throws IOException {
		String layout = null;
		File nnDir = new File(dir+"/"+nnName);
		if(nnDir.exists()){
			try {
				layout = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/layout.txt")));
			} catch (IOException e) {
				throw new RuntimeException("Failed to load neural network "+nnName, e);
			}
		} else {
			String path = findZip(nnName);
			layout = new String(findInZip(path, "layout.txt"));
		}
		return layout;
	}
	
	@Override
	public void storeLayout(String nnName, String layout){
		File l = new File(dir+"/"+nnName+"/layout.txt");

		try(PrintWriter p = new PrintWriter(l)) {
			p.write(layout);
		} catch(Exception e){
			e.printStackTrace();
		}		
	}
	
	@Override
	public Tensor loadParameters(UUID moduleId, String... tag) {
		return load(moduleId, tag);
	}

	@Override
	public Map<UUID, Tensor> loadParameters(Collection<UUID> moduleIds,
			String... tag) {
		return moduleIds.stream().collect(
				Collectors.toMap(moduleId -> moduleId, moduleId -> loadParameters(moduleId, tag)));
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String nnName, String... tag) throws Exception {
		Map<UUID, Tensor> parameters  = new HashMap<>();
		
		NeuralNetworkDTO nn = loadNeuralNetwork(nnName);
		// TODO should we deduce based on ModuleDTO whether the module is trainable and throw
		// exception when trainable module has no parameters on file system?
		for(ModuleDTO m : nn.modules.values()){
			try {
				parameters.put(m.id, load(m.id, tag));
			} catch(Exception e){
				// ignore if no parameters found for a module
			}
		}
		if(parameters.isEmpty())
			throw new Exception("No parameters available for NN "+nnName);
		return parameters;
	}
	
	@Override
	public void storeParameters(UUID nnId, UUID moduleId, Tensor parameters, String... tag) {
		store(moduleId, parameters, tag);
		
		notifyListeners(nnId, Collections.singleton(moduleId), tag);
	}
	
	@Override
	public void storeParameters(UUID nnId, Map<UUID, Tensor> parameters, String... tag) {
		parameters.entrySet().stream().forEach(e -> store(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(parameters.keySet());
		notifyListeners(nnId, uuids, tag);
	}
	
	@Override
	public Set<String> listTags(UUID moduleId){
		Set<String> tags = new TreeSet<>();
		
		File w = new File(dir+"/weights");
		tags.addAll(listTags(moduleId, w));
	
		
		File d = new File(dir);
		for(String l : d.list()){
			File f = new File(dir+"/"+l);
			tags.addAll(listTags(moduleId, f));
		}
		
		return tags;
	}

	@Override
	public Set<String> listTags(String nnName){
		Set<String> tags = new TreeSet<>();
		NeuralNetworkDTO nn = loadNeuralNetwork(nnName);
		nn.modules.keySet().forEach(uuid -> tags.addAll(listTags(uuid)));
		return tags;
	}
	
	private Collection<String> listTags(UUID moduleId, File f){
		Set<String> tags = new TreeSet<>();
		
		if(!f.exists())
			return tags;
		
		List<String> candidates = null;
		if(!f.isDirectory()){
			try(ZipFile zip = new ZipFile(f)){
				candidates = zip.stream().filter(e -> e.getName().startsWith(moduleId.toString())).map(e -> e.getName()).collect(Collectors.toList());
			} catch(Exception e){
				return tags;
			}
		} else {
			candidates = Arrays.asList(f.list(new FilenameFilter() {
				@Override
				public boolean accept(File dir, String name) {
					return name.startsWith(moduleId.toString());
				}
			}));
		}
		if(candidates != null){
			for(String candidate : candidates){
				parseTags(candidate.substring(37), tags);
			}
		}
		
		return tags;
	}
	
	private void parseTags(String candidate, Set<String> result){
		if(candidate.length() >= 36){
			try {
				// correctly parse out UUID as single tags
				UUID uuidAsTag = UUID.fromString(candidate.substring(0, 36));
				result.add(uuidAsTag.toString());
				if(candidate.length() > 37)
					parseTags(candidate.substring(37), result);
				return;
			} catch(Exception e){}
		}
		
		int split = candidate.indexOf('-');
		if(split > 0){
			String tag = candidate.substring(0, split);
			result.add(tag);
			if(candidate.length() > split)
				parseTags(candidate.substring(split+1), result);
			return;
		}
		
		result.add(candidate);
	}
	
	@Override
	public void accParameters(UUID nnId, UUID moduleId, Tensor accParameters, String... tag){
		acc(moduleId, accParameters, tag);
		
		notifyListeners(nnId, Collections.singleton(moduleId), tag);
	}

	@Override
	public void accParameters(UUID nnId, Map<UUID, Tensor> accParameters, String... tag) {
		accParameters.entrySet().stream().forEach(e -> acc(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(accParameters.keySet());
		notifyListeners(nnId, uuids, tag);

	}
	
	@Override
	public long spaceLeft() {
		File d = new File(dir);
		return d.getUsableSpace();
	}
	
	private Tensor load(UUID moduleId, String... tag){
		try {
			lock.read(moduleId);
			
			// first check weights, next check all other nn dirs
			File f = new File(dir+"/weights/"+parametersId(moduleId, tag));
			if(f.exists()){
				try (DataInputStream is = new DataInputStream(
						new BufferedInputStream(new FileInputStream(f)));
				){
					return readTensor(is);
				} catch(IOException e){
				} 
			}
			
			File d = new File(dir);
			for(File dd : d.listFiles()){
				if(dd.isDirectory()){
					f = new File(dir+"/"+dd.getName()+"/"+parametersId(moduleId, tag));
					if(f.exists()){
						try (DataInputStream is = new DataInputStream(
								new BufferedInputStream(new FileInputStream(f)));
						){
							return readTensor(is);
						} catch(IOException e){
						}
					}
				} else {	
					try (
						ZipFile zip = new ZipFile(dd);
						DataInputStream is = new DataInputStream(
								new BufferedInputStream(
									zip.getInputStream(zip.getEntry(parametersId(moduleId, tag)))));
					){
						return readTensor(is);
					} catch(IOException e){
						throw e;
					} 
				}
			}
			throw new FileNotFoundException();
		} catch(Exception e){
			throw new RuntimeException("Failed to load parameters for module "+moduleId+" with tags "+Arrays.toString(tag), e);
		} finally {
			lock.free(moduleId);
		}
	}
	
	private Tensor readTensor(DataInputStream is) throws IOException{
		// load tensor in chunks, slightly slower than one copy from Java to native,
		// but reduces memory usage a lot for big tensors
		int bufferSize = 10000;
		float[] data = new float[bufferSize];
		
		int length = is.readInt();
		Tensor t = new Tensor(length);
		int index = 0;
		while(length > 0){
			if(length<bufferSize){
				bufferSize = length;
				data = new float[bufferSize];
			}
			
			for(int i=0;i<bufferSize;i++){
				data[i] = is.readFloat();
			}
			
			t.narrow(0, index, bufferSize).set(data);;
			
			length -= bufferSize;
			index+= bufferSize;
		}
		return t;
	}
	
	
	private void store(UUID moduleId, Tensor parameters, String... tag){
		try {
			lock.write(moduleId);
			
			File f = new File(dir+"/weights/"+parametersId(moduleId, tag));
	
			try(DataOutputStream os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
				float[] data = parameters.get();
				os.writeInt(data.length);
				for(int i=0;i<data.length;i++){
					os.writeFloat(data[i]);
				}
				os.flush();
				os.close();
			} catch(IOException e){
				e.printStackTrace();
			}
		} catch(InterruptedException e){
			// ignore?
		} finally {
			lock.free(moduleId);
		}
	}
	
	private  void acc(UUID moduleId, Tensor accParameters, String... tag){
		try {
			lock.write(moduleId);
			Tensor parameters = accParameters;
			try {
				parameters = load(moduleId, tag);
				
				TensorOps.add(parameters, parameters, accParameters);
			} catch(Exception e){
				System.out.println("Failed to load parameters for "+moduleId+" "+Arrays.toString(tag)+", store as new");
			}
		
			store(moduleId, parameters, tag);
		} catch(InterruptedException e){
			// ignore?
		} finally {
			lock.free(moduleId);
		}
	}
	
	private String parametersId(UUID id, String[] tag){
		String pid = id.toString();
		if(tag!=null && tag.length>0){
			for(String t : tag){
				if(t!=null)
					pid+="-"+t;
			}
		}
		return pid;
	}
	
	private String findZip(String nnName){
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(!f.isDirectory()){
				String name = f.getName();
				name = name.substring(0, name.indexOf("."));
				if(name.equals(nnName)){
					return f.getAbsolutePath();
				}
			}
		}
		throw new RuntimeException("Failed to find "+nnName);
	}
	
	private byte[] findInZip(String path, String entry){
		// check whether this is a .zip with modules.txt inside...
		try(ZipFile zip = new ZipFile(path)) {
			ZipEntry e = zip.getEntry(entry);
			
			byte[] data = new byte[(int)e.getSize()];
			DataInputStream is = new DataInputStream(zip.getInputStream(e));
			is.readFully(data);
			return data;
		} catch (IOException ex) {
			throw new RuntimeException("Failed to find entry "+entry+" in zip "+path, ex);
		}
			
	}
	
	private void notifyListeners(UUID nnId, Collection<UUID> moduleIds, String... tag){
		synchronized(listeners){
			// match tags and nnId
			List<String> tags = new ArrayList<String>(tag.length+1);
			tags.addAll(Arrays.asList(tag));
			tags.add(nnId.toString());
			final List<RepositoryListener> toNotify = listeners.entrySet()
					.stream()
					.filter( e -> match(e.getValue(), moduleIds, tags))
					.map( e -> e.getKey())
					.collect(Collectors.toList());
			
			executor.execute( ()->{
				for(RepositoryListener l : toNotify){
					l.onParametersUpdate(nnId, moduleIds, tag);
				}
			});
			
		}
	}
	
	private boolean match(Collection<String> targets, Collection<UUID> moduleIds, List<String> tags){
		// match everything if targets = null
		if(targets==null){
			return true;
		}
		
		// targets in form  moduleId:tag
		for(String target : targets){
			String[] split = target.split(":");
			if(split[0].length()!=0){
				// moduleId provided
				if(!moduleIds.contains(UUID.fromString(split[0]))){
					return false;
				}
			}
			
			// some tag provided
			for(int i=1;i<split.length;i++){
				String t = split[i];
				if(tags!=null){
					if(!tags.contains(t)){
						return false;
					}
				} else {
					return false;
				}
			}
			return true;
		}
		return false;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addRepositoryListener(RepositoryListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			listeners.put(l, Arrays.asList(targets));
		} else {
			listeners.put(l, null);
		}
	}
	
	void removeRepositoryListener(RepositoryListener l){
		listeners.remove(l);
	}

}
