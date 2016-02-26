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
		
		Thread t = new Thread(()->checkDir());
		t.start();
		
	}
	
	@Reference
	void setConfigurationAdmin(ConfigurationAdmin ca){
		this.configAdmin = ca;
	}
	
	private void checkDir(){
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
						properties.put(pair[0].trim(), pair[1].trim());
					}
					
					properties.put("dir", poolDir);

					String factoryPid = FileExperiencePool.class.getName();
					String instancePid = configAdmin.createFactoryConfiguration(factoryPid, null).getPid();
					configAdmin.getConfiguration(instancePid).update(properties);
				
				} catch(IOException e){
					System.out.println("Error initializing experience pool "+s);
					e.printStackTrace();
				} finally {
					try {
						input.close();
					} catch (IOException e) {}
				}
			}
		}
	}
}
