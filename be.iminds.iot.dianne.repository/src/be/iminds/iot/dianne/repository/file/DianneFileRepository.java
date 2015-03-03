package be.iminds.iot.dianne.repository.file;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
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
		try {
			DataOutputStream os = new DataOutputStream(new FileOutputStream(f));
			os.writeInt(weights.length);
			for(int i=0;i<weights.length;i++){
				os.writeFloat(weights[i]);
			}
			os.flush();
			os.close();
		} catch(IOException e){
			e.printStackTrace();
		}
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
}
