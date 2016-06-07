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
package be.iminds.iot.dianne.rl.ale;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.ale.config.ALEConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Arcade Learning Environment for learning agents to play Atari games
 *  
 * @author tverbele
 *
 */
@Component(immediate = true,
property = { "name="+ArcadeLearningEnvironment.NAME, "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class ArcadeLearningEnvironment implements Environment {

	public static final String NAME = "ALE";
	
    static {
		try {
		    System.loadLibrary("ALE");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library ALE failed to load. \n"+ e);
		    throw e;
		}
    }

	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());

	private ALEConfig config;
	private volatile boolean active = false;
	
	private Tensor observation;
	// placeholder tensors for generating observation
	private Tensor screen; 
	private Tensor gray;
	
    
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		int r = 0;
		
		for(int i=0;i<config.observationLength;i++){
			r += performAction(TensorOps.argmax(action));
			
			screen.set(getScreen());
			
			if(config.grayscale){
				screen.select(0, 0).copyInto(gray);
				TensorOps.add(gray, gray, screen.select(0, 1));
				TensorOps.add(gray, gray, screen.select(0, 2));
				TensorOps.div(gray, gray, 3);
				
				gray.copyInto(observation.select(0, i));
			} else {
				screen.select(0, 0).copyInto(observation.select(0, 3*i));
				screen.select(0, 1).copyInto(observation.select(0, 3*i+1));
				screen.select(0, 2).copyInto(observation.select(0, 3*i+2));
			}
		}
		
    	final float reward = r;
		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
    	
		return r > 0 ? 1 : r < 0 ? -1 : 0;
	}

	@Override
	public Tensor getObservation(Tensor t) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		if(gameOver()){
			return null;
		} else {
			return observation.copyInto(t);
		}
	}

	@Override
	public void reset() {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		resetGame();
		
		screen.set(getScreen());
		if(config.grayscale){
			screen.select(0, 0).copyInto(gray);
			TensorOps.add(gray, gray, screen.select(0, 1));
			TensorOps.add(gray, gray, screen.select(0, 2));
			TensorOps.div(gray, gray, 3);
				
			for(int i=0;i<config.observationLength;i++){
				gray.copyInto(observation.select(0, i));
			}
		} else {
			for(int i=0;i<config.observationLength;i++){
				screen.select(0, 0).copyInto(observation.select(0, 3*i));
				screen.select(0, 1).copyInto(observation.select(0, 3*i+1));
				screen.select(0, 2).copyInto(observation.select(0, 3*i+2));
			}
		}
	}

	private native void loadROM(String rom);

	private native int getActions();
	
	private native int performAction(int index);
	
	private native void resetGame();
	
	private native boolean gameOver();
	
	private native float[] getScreen();
	
	private native void setFrameskip(int skip);
	
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target == null || target.equals(NAME)){
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
		
		this.config = DianneConfigHandler.getConfig(config, ALEConfig.class);
		
		// init screen tensor
    	screen = new Tensor(3, 210, 160);
    	if(this.config.grayscale){
    		gray = new Tensor(210, 160);
    	}
		int channels = this.config.grayscale ? 1 : 3;
		observation = new Tensor(this.config.observationLength*channels, 210, 160);

    	
    	loadROM(this.config.rom);
    	setFrameskip(this.config.skip);

    	active = true;
    	
    	reset();
	}

	@Override
	public void cleanup() {
		//TODO cleanup?
		active = false;
	}
}
