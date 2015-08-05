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
import java.util.List;
import java.util.Map.Entry;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

@Component(immediate=true)
public class DianneFileRepository implements DianneRepository {

	private String dir = "nn";
	
	private final JsonParser parser = new JsonParser();
	
	@Activate
	public void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.storage");
		if(s!=null){
			dir = s;
		}
		File d = new File(dir+"/weights/");
		d.mkdirs();
	}

	@Override
	public List<String> avialableNeuralNetworks() {
		List<String> networks = new ArrayList<String>();
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(f.isDirectory()){
				String name = f.getName();
				if(!name.equals("weights")){
					networks.add(f.getName());
				}
			}
		}
		return networks;
	}

	@Override
	public NeuralNetworkDTO loadNeuralNetwork(String network) throws IOException {
		String nn = new String(Files.readAllBytes(Paths.get(dir+"/"+network+"/modules.txt")));
		return DianneJSONConverter.parseJSON(nn);
	}
	
	@Override
	public void storeNeuralNetwork(String network, String modules){
		File d = new File(dir+"/"+network);
		d.mkdirs();
		
		File n = new File(dir+"/"+network+"/modules.txt");
		
		// also look for weights and move these to the network folder
		JsonObject json = (JsonObject)parser.parse(modules);
		for(Entry<String, JsonElement> e : json.entrySet()){
			File weights = new File(dir+"/weights/"+e.getKey());
			if(weights.exists()){
				// move to network folder
				weights.renameTo(new File(dir+"/"+network+"/"+e.getKey()));
			}
		}
		
		Gson gson = new GsonBuilder().setPrettyPrinting().create();
		String output = gson.toJson(json);
		
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
	public String loadLayout(String network) throws IOException {
		String layout = new String(Files.readAllBytes(Paths.get(dir+"/"+network+"/layout.txt")));
		return layout;
	}
	
	@Override
	public void storeLayout(String network, String layout){
		File l = new File(dir+"/"+network+"/layout.txt");
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
	public float[] loadWeights(UUID id) throws IOException {
		File d = new File(dir);
		File f = null;
		for(String l : d.list()){
			f = new File(dir+"/"+l+"/"+id.toString());
			if(f.exists()){
				break;
			} else {
				f = null;
			}
		}
		if(f!=null){
			DataInputStream is = new DataInputStream(new BufferedInputStream(new FileInputStream(f)));
			int count = is.readInt();
			float[] weights = new float[count];
			for(int i=0;i<count;i++){
				weights[i] = is.readFloat();
			}
			is.close();
			return weights;
		}
		throw new FileNotFoundException();
	}

	@Override
	public void storeWeights(UUID id, float[] weights) {
		File f = new File(dir+"/weights/"+id.toString());
		DataOutputStream os = null;
		try {
			os = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)));
			os.writeInt(weights.length);
			for(int i=0;i<weights.length;i++){
				os.writeFloat(weights[i]);
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

}
