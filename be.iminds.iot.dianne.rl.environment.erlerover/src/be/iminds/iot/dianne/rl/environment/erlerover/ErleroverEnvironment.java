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
package be.iminds.iot.dianne.rl.environment.erlerover;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeoutException;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.Configuration;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.rover.Rover;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.simulator.api.Simulator;

/**
 * Environment to race around with the Erlerover
 * 
 * @author tverbele
 *
 */
@Component(immediate = true,
	property = { "name="+ErleroverEnvironment.NAME, "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class ErleroverEnvironment implements Environment {
	
	public static final String NAME = "Rover";
	
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());
	
	private Tensor observation; 

	private volatile Rover rover;
	private volatile LaserScanner laser;
	private volatile Simulator simulator;
	
	private ConfigurationAdmin ca;
	private Configuration laserConfig;
	private Configuration roverConfig;
	
	
	private int scanPoints = 128;
	
	private boolean terminal = false;
	private volatile boolean active = false;
	
	private Process process;
	
	@Activate
	void activate(BundleContext context) throws Exception {
		boolean headless = false;
		String h = context.getProperty("gazebo.headless");
		if(h != null){
			headless = Boolean.parseBoolean(h);
		}
		
		try {
			// unpack the models into the current dir
			Enumeration<URL> urls = context.getBundle().findEntries("scenes", "*.sdf", true);
			while(urls.hasMoreElements()){
				URL url = urls.nextElement();
				
				// make scenes directory
				File dir = new File("scenes");
				dir.mkdirs();
				
				try(InputStream inputStream = url.openStream();
					OutputStream outputStream =
							new FileOutputStream(new File(url.getFile().substring(1)));){
					int read = 0;
					byte[] bytes = new byte[1024];
	
					while ((read = inputStream.read(bytes)) != -1) {
						outputStream.write(bytes, 0, read);
					}
				}
			}
		} catch(Exception e){
			e.printStackTrace();
		}
		
		// initiate gazebo process?
		try {
			List<String> cmd = new ArrayList<>();
			cmd.add("roslaunch");
			cmd.add("ardupilot_sitl_gazebo_plugin");
			cmd.add("rover_spawn.launch");
			cmd.add("gui:="+!headless);
			ProcessBuilder builder = new ProcessBuilder(cmd);
			builder.inheritIO();
			process = builder.start();
		} catch(Exception e){
			System.err.println("Error launching native gazebo");
			throw e;
		}
		
		// add configurations for rover and laserscanner
		laserConfig = ca.createFactoryConfiguration("be.iminds.iot.sensor.range.ros.LaserScanner", null);
		Dictionary<String, String> laserDict = new Hashtable<>();
		laserDict.put("name", "hokuyo");
		laserDict.put("topic", "/scan");
		laserConfig.update(laserDict);
		
		roverConfig = ca.createFactoryConfiguration("be.iminds.iot.robot.erlerover.ros.Rover", null);
		Dictionary<String, String> roverDict = new Hashtable<>();
		roverDict.put("name", "rover");
		roverConfig.update(roverDict);

	}
	
	@Deactivate
	void deactivate(){
		if(process!=null){
			process.destroy();
		}
		
		if(laserConfig != null)
			try {
				laserConfig.delete();
			} catch (IOException e) {
			}
		
		if(roverConfig != null)
			try {
				roverConfig.delete();
			} catch (IOException e) {
			}
	}
	
	@Override
	public int[] observationDims() {
		return new int[]{scanPoints};
	}

	@Override
	public int[] actionDims() {
		return new int[]{4};
	}
	
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");

		float reward;
		
		// get throttle/yaw from action
		float throttle = 0;
		float yaw = 0;
		
		// apply action - for now discrete TODO split up with continuous variant
		int a = TensorOps.argmax(action);
		switch(a){
		case 0:
			// forward
			throttle = 1;
			yaw = 0;
			break;
		case 1:
			// stop
			throttle = 0;
			yaw = 0;
			break;
		case 2:
			// left
			throttle = 1;
			yaw = -1;
			break;
		case 3:
			// right
			throttle = 1;
			yaw = 1;
			break;
		}

		rover.move(throttle, yaw);
		
		try {
			simulator.tick();
		} catch (TimeoutException e) {
			throw new RuntimeException("The Environment timed out!");
		}
		
		// update observation
		updateObservation();
		
		// get reward
		float min = TensorOps.min(observation);
		
		if(min < 0.4){
			reward = -1;
			terminal = true;
		} else {
			reward = throttle;
		}
		
		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
			
		return reward;
	}
	
	@Override
	public Tensor getObservation(Tensor t) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		if(terminal)
			return null;

		return observation.copyInto(t);
	}

	@Override
	public void reset() {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		// reset
		simulator.stop();
		rover.stop();
		terminal = false;
		
		simulator.start(true);
		
		try {
			simulator.tick();
		} catch (TimeoutException e) {
			throw new RuntimeException("The Environment timed out!");
		}
		
		updateObservation();
		scanPoints = observation.size();
		
		listeners.stream().forEach(l -> l.onAction(0, observation));
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target==null || target.equals(NAME)){
			listeners.add(l);
		}
	}
	
	void removeEnvironmentListener(EnvironmentListener l){
		listeners.remove(l);
	}

	@Override
	public void setup(Map<String, String> config) {
		if(active)
			throw new RuntimeException("This Environment is already active");
		
		active = true;

		while(simulator == null && active){
			try {
				Thread.sleep(1000); //TODO timeout
			} catch (InterruptedException e) {
				return;
			}
		}

		this.simulator.start(false);

		while((rover == null
				|| laser == null)
				&& active){
			try {
				Thread.sleep(1000); //TODO timeout
			} catch (InterruptedException e) {
				return;
			}
		}
				
		reset();
	}

	@Override
	public void cleanup() {
		active = false;
	}
	
	
	private void updateObservation(){
		float[] data = laser.getValue().data;
		float[] result = Arrays.copyOf(data, data.length);
		observation = new Tensor(result, data.length);
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setRover(Rover r){
		this.rover = r;
	}
	
	void unsetRover(Rover r){
		if(this.rover==r)
			this.rover = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setLaserScanner(LaserScanner l){
		this.laser = l;
	}
	
	void unsetLaserScanner(LaserScanner l){
		if(this.laser==l)
			this.laser = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setSimulator(Simulator s){
		this.simulator = s;
		Map<String, String> config = new HashMap<>();
		config.put("pose", "2.4, -2.1, 0");
		this.simulator.loadScene("scenes/circuit1.sdf", config);
	}
	
	void unsetSimulator(Simulator s){
		if(this.simulator == s){
			this.simulator = s;
		}
	}
	
	@Reference
	void setConfigurationAdmin(ConfigurationAdmin ca){
		this.ca = ca;
	}
}
