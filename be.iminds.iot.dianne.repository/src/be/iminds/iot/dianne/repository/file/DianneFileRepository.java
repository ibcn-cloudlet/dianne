package be.iminds.iot.dianne.repository.file;

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
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.repository.DianneRepository;

@Component(immediate=true)
public class DianneFileRepository implements DianneRepository {

	private String dir = "nn";
	
	@Activate
	public void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.storage");
		if(s!=null){
			dir = s+"";
		}
		File d = new File(dir);
		d.mkdirs();
	}

	@Override
	public List<String> networks() {
		List<String> networks = new ArrayList<String>();
		File d = new File(dir);
		for(File f : d.listFiles()){
			if(f.isDirectory()){
				networks.add(f.getName());
			}
		}
		return networks;
	}

	@Override
	public String loadNetwork(String network) throws IOException {
		String modules = new String(Files.readAllBytes(Paths.get(dir+"/"+network+"/modules.txt")));
		return modules;
	}
	
	@Override
	public void storeNetwork(String network, String modules){
		File n = new File(dir+"/"+network+"/modules.txt");
		PrintWriter p = null;
		try {
			p = new PrintWriter(n);
			p.write(modules);
		} catch(Exception e){
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
		} finally{
			if(p!=null){
				p.close();
			}
		}		
	}
	
	@Override
	public float[] loadWeights(UUID id) throws IOException {
		File f = new File(dir+"/weights/"+id.toString());
		if(f.exists()){
			DataInputStream is = new DataInputStream(new FileInputStream(f));
			int count = is.readInt();
			float[] weights = new float[count];
			for(int i=0;i<count;i++){
				weights[i] = is.readFloat();
			}
			is.close();
			return weights;
		}
		throw new FileNotFoundException(f.getAbsolutePath());
	}

	@Override
	public void storeWeights(UUID id, float[] weights) {
		File f = new File(dir+"/weights/"+id.toString());
		DataOutputStream os = null;
		try {
			os = new DataOutputStream(new FileOutputStream(f));
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
