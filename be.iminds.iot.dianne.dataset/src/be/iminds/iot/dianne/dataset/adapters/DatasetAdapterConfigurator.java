package be.iminds.iot.dianne.dataset.adapters;

import java.io.File;
import java.io.FileReader;
import java.util.Hashtable;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.Configuration;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

@Component(immediate=true)
public class DatasetAdapterConfigurator {

	private ConfigurationAdmin ca;
	
	private String path = "datasets";
	
	@Activate
	void activate(BundleContext context) {
		String d = context.getProperty("be.iminds.iot.dianne.datasets.location");
		if(d != null){
			path = d;
		}
		
		File dir = new File(path);
		if(!dir.isDirectory())
			return;
		
		for(File f : dir.listFiles()){
			try {
				// parse any adapter configurations from JSON and apply config?
				JsonParser parser = new JsonParser();
				JsonObject json = parser.parse(new JsonReader(new FileReader(f))).getAsJsonObject();
				
				String pid = json.get("adapter").getAsString();
				if(pid != null){
					String dataset = json.get("dataset").getAsString();
					
					// TODO use object conversion from JSON here?
					Configuration config = ca.createFactoryConfiguration(pid);
					Hashtable<String, Object> props = new Hashtable<>();
					props.put("Dataset.target", "(name="+dataset+")");
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
				}
			} catch(Exception e){
				// ignore?!
			}
		}
	}
	
	@Reference
	void setConfigurationAdmin(ConfigurationAdmin ca){
		this.ca = ca;
	}
}
