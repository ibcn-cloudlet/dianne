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
package be.iminds.iot.dianne.rl.environment.ale;

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
import be.iminds.iot.dianne.rl.environment.ale.config.ALEConfig;
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
	private Tensor scaled;
	private Tensor narrowed;
	
	@Override
	public int[] observationDims() {
		return new int[]{this.config.observationLength, 84, 84};
	}

	@Override
	public int[] actionDims() {
		return new int[]{getActions()};
	}
	
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		int r = 0;
		
		for(int i=0;i<config.observationLength;i++){
			r += performAction(TensorOps.argmax(action));
			
			screen.set(getScreen());
			
			// convert to grayscale
			gray = screen.select(0, 0).copyInto(gray);
			TensorOps.add(gray, gray, screen.select(0, 1));
			TensorOps.add(gray, gray, screen.select(0, 2));
			TensorOps.div(gray, gray, 3);
			// downsample to 110 x 84
			TensorOps.scale2D(scaled, gray, 110, 84);
			// copy narrowed 84x84 into observation
			narrowed.copyInto(observation.select(0, i));
		}
		
    	final float reward = r;
		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, screen));
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

		screen.select(0, 0).copyInto(gray);
		TensorOps.add(gray, gray, screen.select(0, 1));
		TensorOps.add(gray, gray, screen.select(0, 2));
		TensorOps.div(gray, gray, 3);
				
		TensorOps.scale2D(scaled, gray, 110, 84);

		for(int i=0;i<config.observationLength;i++){
			narrowed.copyInto(observation.select(0, i));
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
    	gray = new Tensor(210, 160);
		
    	scaled = new Tensor(110, 84);
    	// narrow with offset in y-dim to crop game area
    	narrowed = scaled.narrow(0, 18, 84);
    	
    	observation = new Tensor(this.config.observationLength, 84, 84);

    	// parameters should be set before ROM is loaded!
    	setFrameskip(this.config.skip);
    	
    	loadROM(this.config.rom);

    	active = true;
    	
    	reset();
	}

	@Override
	public void cleanup() {
		//TODO cleanup?
		active = false;
	}
}
