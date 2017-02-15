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
package be.iminds.iot.dianne.rl.environment.kuka;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.KukaConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.robot.api.OmniDirectional;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.simulator.api.Simulator;

/**
 * Abstract Kuka environment that can be used as base for running the Kuka with
 * laser range scanner in our cage
 * 
 * In case no simulator is available, the environment will pause after each termination.
 * 
 * Make sure to configure the properties in your inheriting components:
 * 
@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+<your environment name>, 
				 "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment",
				 "aiolos.combine=*",
				 "osgi.command.scope=kukaagent",
				 "osgi.command.function=end",
				 "osgi.command.function=pause",
				 "osgi.command.function=go",
				 "osgi.command.function=reward",
				 "osgi.command.function=load"})
 * 
 * @author tverbele
 *
 */
public abstract class AbstractKukaEnvironment implements Environment, KukaEnvironment {
	
	protected KukaConfig config;

	private Map<String, String> configMap;
	
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());
	
	protected volatile boolean active = false;
	protected boolean terminal = false;
	protected Tensor observation;
	protected Tensor noise;
	
	// TODO for now limited to 1 youbot
	protected OmniDirectional kukaPlatform;
	protected Arm kukaArm;
	protected SortedMap<String,LaserScanner> rangeSensors;
	
	// Environment can be both simulated or on real robot
	protected Simulator simulator;
	
	protected float reward = 0;
	private volatile boolean pause = false;
	
	public abstract String getName();
	
	@Activate
	void activate(BundleContext context) {
		try {
			// unpack the scenes into the current dir
			Enumeration<URL> urls = context.getBundle().findEntries("scenes", "*.ttt", true);
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
	}
	
	@Override
	public int[] observationDims() {
		// observation from laser range scanner
		return this.rangeSensors.size() > 1 ? new int[]{this.rangeSensors.size(), config.scanPoints} : new int[]{config.scanPoints};
	}

	@Override
	public abstract int[] actionDims();
	
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		// execute action
		try {
			executeAction(action);
		} catch (Exception e) {
			throw new RuntimeException("Failed executing action "+action, e);
		}
		
		// calculate reward
		try {
			reward = calculateReward();
			reward -= config.energyPenalization*calculateEnergy(action);
		} catch(Exception e){
			throw new RuntimeException("Failed calculating reward");
		}
		
		if(simulator == null && terminal){
			// pause the agent on termination to enter reward
			pause = true;
			System.out.println("Enter your reward for this episode (type \"reward x\" in CLI with x your reward as floating point)");
		}
		
		// pause in case one wants to set reward
		if(pause){
			kukaPlatform.stop();
			waitForResume();
		}
		
		updateObservation();

		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
			
		return reward;
	}
	
	@Override
	public Tensor getObservation(Tensor t) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		if(terminal){
			return null;
		}
		
		return observation.copyInto(t);
	}

	@Override
	public void reset() {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		terminal = false;
		
		// TODO handle failure here?
		try {
			deinit();

			init();
		} catch(Exception e){
			throw new RuntimeException("Failed to initialize the environment ...", e);
		}
		
		updateObservation();
		
		listeners.stream().forEach(l -> l.onAction(0, observation));
	}
	
	
	protected abstract void executeAction(Tensor a) throws Exception; 
	
	protected abstract float calculateReward() throws Exception;
	
	protected float calculateEnergy(Tensor a) throws Exception {
		return 1;
	}
	
	protected abstract void initSimulator() throws Exception;
	
	protected abstract void configure(Map<String, String> config);
		
	public void reward(float r){
		this.reward = r;
		resume();
	}
	
	public void stop(){
		cleanup();
	}
	
	public void pause(){
		pause = true;
	}
	
	public void resume(){
		pause = false;
		synchronized(this){
			this.notifyAll();
		}
	}
	
	public void start(String... params){
		HashMap<String, String> config = new HashMap<>();
		config.put("tick", "false");
		for(String p : params){
			if(p.contains("=")){
				String[] keyval = p.split("=");
				config.put(keyval[0], keyval[1]);
			}
		}
		setup(config);
	}
	
	private void waitForResume(){
		synchronized(this){
			if(pause){
				try {
					this.wait();
				} catch (InterruptedException e) {
					System.out.println("Environment interrupted!");
				}
				pause = false;
			}
		}
	}
	
	private void updateObservation(){
		int totalLength = 0;
		synchronized (rangeSensors) {
			for (LaserScanner ls : rangeSensors.values()) {
				totalLength += ls.getValue().data.length;
			}
			try {
				Iterator<LaserScanner> it = rangeSensors.values().iterator();
				float[] data = it.next().getValue().data; // thorws NoSuchElementException on empty list
				float[] result = Arrays.copyOf(data, totalLength);
				int offset = data.length;
				int dimension = data.length;
				while(it.hasNext()) {
					data = it.next().getValue().data;
					System.arraycopy(data, 0, result, offset, data.length);
				    offset += data.length;
				}
				observation = new Tensor(result, rangeSensors.size(), dimension);
				if(config.rangeSensorNoise > 0) {
					if(noise == null || !noise.sameDim(observation))
						noise = new Tensor(observation.dims());
					
					noise.randn();
					TensorOps.add(observation, observation, config.rangeSensorNoise, noise);
				}
			} catch (NoSuchElementException ex) {}
		}
	}
	
	
	private void init() throws Exception {
		if(simulator == null) {
			// automatically pause the environment until the user resumes from CLI
			pause = true;
			System.out.println("Reset your environment and resume by typing the \"start\" command.");
			waitForResume();
			
		} else {
			try {
				initSimulator();
			} catch(Exception e){
				// try to kill the simulator?! - this is hacky!
				// TODO this should be fixed in the robot project?
				Runtime.getRuntime().exec("killall vrep");
                simulator = null;
            	System.out.println("Unexpected simulator error, waiting for simulator to come back online...");
				long start = System.currentTimeMillis();
                while(simulator == null){
                	if(System.currentTimeMillis()-start > config.timeout){
                      	throw new Exception("Failed to restart simulator");
                    }
                	
                	Thread.sleep(1000);
                }
                
                // configure it again from scratch
                configure(configMap);
                
                initSimulator();
			}
		}
	}
	
	private void deinit(){
		if(simulator != null){
			simulator.stop();
			// TODO wait until simulation has completely stopped (use function call from remote api?)
			try {
				Thread.sleep(config.wait);
			} catch (InterruptedException e) {}
		}
	}
	

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target==null || target.equals(getName())){
			listeners.add(l);
		}
	}
	
	void removeEnvironmentListener(EnvironmentListener l){
		listeners.remove(l);
	}

	
	// TODO use target filters for these  (involves spawning environments from configadmin?)
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setArm(Arm a){
		this.kukaArm = a;
	}
	
	void unsetArm(Arm a){
		this.kukaArm = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setPlatform(OmniDirectional o){
		this.kukaPlatform = o;
	}
	
	void unsetPlatform(OmniDirectional o){
		if(this.kukaPlatform==o)
			this.kukaPlatform = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, policy=ReferencePolicy.DYNAMIC)
	void bindLaserScanner(LaserScanner l, Map<String,String> config){
		if (this.rangeSensors == null)
			rangeSensors = Collections.synchronizedSortedMap(new TreeMap<>());
		this.rangeSensors.put(config.get("name"), l);
	}
	
	void unbindLaserScanner(LaserScanner l, Map<String,String> config){
		this.rangeSensors.remove(config.get("name"));
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setSimulator(Simulator s){
		this.simulator = s;
	}
	
	void unsetSimulator(Simulator s){
		if(this.simulator == s){
			this.simulator = s;
		}
	}
	
	@Override
	public void setup(Map<String, String> config) {
		if(active)
			throw new RuntimeException("This Environment is already active");
		
		this.config = DianneConfigHandler.getConfig(config, KukaConfig.class);
		this.configMap = config;
		
		configure(config);
		
		active = true;
		
		reset();
	}
	
	@Override
	public void cleanup() {
		active = false;
		
		deinit();
	}
}
