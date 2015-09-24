package be.iminds.iot.dianne.repository.file;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
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
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

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
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.google.gson.JsonParser;

@Component(immediate=true)
public class DianneFileRepository implements DianneRepository {

	private String dir = "nn";
	
	private final JsonParser parser = new JsonParser();
	
	private TensorFactory factory;
	
	private Map<RepositoryListener, List<String>> listeners = Collections.synchronizedMap(new HashMap<RepositoryListener, List<String>>());
	protected ExecutorService executor = Executors.newSingleThreadExecutor();

	
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
			if(f.isDirectory()){
				String name = f.getName();
				if(!name.equals("weights")){
					nns.add(f.getName());
				}
			}
		}
		return nns;
	}

	@Override
	public NeuralNetworkDTO loadNeuralNetwork(String nnName){
		try {
			String nn = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/modules.txt")));
			return DianneJSONConverter.parseJSON(nn);
		} catch (IOException e) {
			throw new RuntimeException("Failed to load neural network "+nnName, e);
		}
	}
	
	@Override
	public void storeNeuralNetwork(NeuralNetworkDTO nn){
		File d = new File(dir+"/"+nn.name);
		d.mkdirs();
		
		File n = new File(dir+"/"+nn.name+"/modules.txt");
		
		String output = DianneJSONConverter.toJsonString(nn, true);
		
		PrintWriter p = null;
		try {
			p = new PrintWriter(n);
			p.write(output);
		} catch(Exception e){
			e.printStackTrace();
		} finally{
			if(p!=null){
				p.close();
			}
		}
	}

	@Override
	public String loadLayout(String nnName) throws IOException {
		String layout = new String(Files.readAllBytes(Paths.get(dir+"/"+nnName+"/layout.txt")));
		return layout;
	}
	
	@Override
	public void storeLayout(String nnName, String layout){
		File l = new File(dir+"/"+nnName+"/layout.txt");
		PrintWriter p = null;
		try {
			p = new PrintWriter(l);
			p.write(layout);
		} catch(Exception e){
			e.printStackTrace();
		} finally{
			if(p!=null){
				p.close();
			}
		}		
	}
	
	@Override
	public synchronized Tensor loadParameters(UUID moduleId, String... tag) {
		return load(moduleId, tag);
	}

	@Override
	public synchronized Map<UUID, Tensor> loadParameters(Collection<UUID> moduleIds,
			String... tag) {
		return moduleIds.stream().collect(
				Collectors.toMap(moduleId -> moduleId, moduleId -> loadParameters(moduleId, tag)));
	}

	@Override
	public Map<UUID, Tensor> loadParameters(String nnName, String... tag) {
		Map<UUID, Tensor> parameters  = new HashMap<>();
		
		NeuralNetworkDTO nn = loadNeuralNetwork(nnName);
		for(ModuleDTO m : nn.modules){
			try {
				parameters.put(m.id, load(m.id, tag));
			} catch(Exception e){}
		}
		return parameters;
	}
	
	private Tensor load(UUID moduleId, String... tag){
		try {
			File d = new File(dir);
			File f = null;
			for(String l : d.list()){
				f = new File(dir+"/"+l+"/"+parametersId(moduleId, tag));
				if(f.exists()){
					break;
				} else {
					f = null;
				}
			}
			if(f!=null){
				DataInputStream is = new DataInputStream(new BufferedInputStream(new FileInputStream(f)));
				int length = is.readInt();
				float[] data = new float[length];
				for(int i=0;i<length;i++){
					data[i] = is.readFloat();
				}
				is.close();
				return factory.createTensor(data, new int[]{length});
			}
			throw new FileNotFoundException();
		} catch(Exception e){
			throw new RuntimeException("Failed to load parameters for module "+moduleId+" with tags "+Arrays.toString(tag), e);
		}
	}
	
	@Override
	public synchronized void storeParameters(UUID moduleId, Tensor parameters, String... tag) {
		store(moduleId, parameters, tag);
		
		notifyListeners(Collections.singleton(moduleId), tag);
	}
	
	@Override
	public synchronized void storeParameters(Map<UUID, Tensor> parameters, String... tag) {
		parameters.entrySet().stream().forEach(e -> store(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(parameters.keySet());
		notifyListeners(uuids, tag);
	}
	
	@Override
	public synchronized void accParameters(UUID moduleId, Tensor accParameters, String... tag){
		acc(moduleId, accParameters, tag);
		
		notifyListeners(Collections.singleton(moduleId), tag);
	}

	@Override
	public synchronized void accParameters(Map<UUID, Tensor> accParameters, String... tag) {
		accParameters.entrySet().stream().forEach(e -> acc(e.getKey(), e.getValue(), tag));
		
		List<UUID> uuids = new ArrayList<UUID>();
		uuids.addAll(accParameters.keySet());
		notifyListeners(uuids, tag);

	}
	
	private void store(UUID moduleId, Tensor parameters, String... tag){
		File f = new File(dir+"/weights/"+parametersId(moduleId, tag));
		DataOutputStream os = null;
		try {
			os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)));
			float[] data = parameters.get();
			os.writeInt(data.length);
			for(int i=0;i<data.length;i++){
				os.writeFloat(data[i]);
			}
			os.flush();
			os.close();
		} catch(IOException e){
			e.printStackTrace();
		} finally {
			if(os!=null){
				try {
					os.close();
				} catch (IOException e) {}
			}
		}
	}
	
	private void acc(UUID moduleId, Tensor accParameters, String... tag){
		Tensor parameters = accParameters;
		try {
			parameters = load(moduleId, tag);
			
			factory.getTensorMath().add(parameters, parameters, accParameters);
		} catch(Exception e){
			System.out.println("Failed to load parameters for "+moduleId+" "+Arrays.toString(tag)+", store as new");
		}
	
		store(moduleId, parameters, tag);
	}
	
	private String parametersId(UUID id, String[] tag){
		String pid = id.toString();
		if(tag!=null && tag.length>0){
			for(String t : tag){
				pid+="-"+t;
			}
		}
		return pid;
	}
	
	private void notifyListeners(Collection<UUID> moduleIds, String... tag){
		synchronized(listeners){
			final List<RepositoryListener> toNotify = listeners.entrySet()
					.stream()
					.filter( e -> match(e.getValue(), moduleIds, tag))
					.map( e -> e.getKey())
					.collect(Collectors.toList());
			
			executor.execute( ()->{
				for(RepositoryListener l : toNotify){
					l.onParametersUpdate(moduleIds, tag);
				}
			});
			
		}
	}
	
	private boolean match(Collection<String> targets, Collection<UUID> moduleIds, String[] tag){
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
				if(tag!=null){
					List<String> tagList = Arrays.asList(tag);
					if(!tagList.contains(t)){
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
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
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
