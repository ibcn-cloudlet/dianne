package be.iminds.iot.dianne.rl.exp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Dictionary;
import java.util.Hashtable;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

@Component(immediate=true)
public class ExperiencePoolLoader {

	private ConfigurationAdmin configAdmin;
	
	private String dir = "exp";
	
	@Activate
	public void activate(BundleContext context) throws Exception {
		String expDir = context.getProperty("be.iminds.iot.dianne.pool");
		if(expDir!=null){
			dir = expDir;
		}
		
		String factoryPid = FileExperiencePool.class.getName();
		
		File d = new File(dir);
		if(d.isDirectory()){
			for(String s : d.list()){
				String poolDir = d.getAbsolutePath()+File.separator+s;
				String propertiesFileName = poolDir+File.separator+"config.properties";
			
				File propertiesFile = new File(propertiesFileName);
				if(!propertiesFile.exists()){
					System.out.println("No config.properties for experience pool "+s);
					continue;
				}
				
				Dictionary<String, Object> properties = new Hashtable<>();

				BufferedReader input = null;
				try {
					input = new BufferedReader(new InputStreamReader(new FileInputStream(propertiesFile)));
					String line = null;
					while((line = input.readLine()) != null){
						if(line.startsWith("#"))
							continue;
						
						if(!line.contains("="))
							continue;
						
						String[] pair = line.split("=");
						
						String key = pair[0].trim();
						
						switch(key){
						case "labels":
							properties.put("labels", pair[1].split(","));
							break;
						case "stateSize":
							properties.put("stateSize", Integer.parseInt(pair[1]));
							break;
						case "actionSize":
							properties.put("actionSize", Integer.parseInt(pair[1]));							
							break;
						default:
							properties.put(pair[0].trim(), pair[1].trim());
							break;
						}
						
					}
					
					properties.put("dir", poolDir);

					String instancePid = configAdmin.createFactoryConfiguration(factoryPid, null).getPid();
					configAdmin.getConfiguration(instancePid).update(properties);
				
				} catch(IOException e){
					System.out.println("Error initializing experience pool "+s);
					e.printStackTrace();
				} finally {
					input.close();
				}
			}
		}
	}
	
	@Reference
	public void setConfigurationAdmin(ConfigurationAdmin ca){
		this.configAdmin = ca;
	}
}
