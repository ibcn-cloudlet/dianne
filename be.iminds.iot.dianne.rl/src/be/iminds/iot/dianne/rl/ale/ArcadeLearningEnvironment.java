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

import java.io.File;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.api.rl.EnvironmentListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

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

    private TensorFactory factory;
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());

	private String rom = "roms/pong.bin"; 
	private int skip = 1;
	private int observationLength = 1; // number of frames in an observation
	private Tensor observation;
    
    @Activate
    public void activate(BundleContext context) throws Exception {
    	String r = context.getProperty("be.iminds.iot.dianne.rl.ale.rom");
    	if(r!=null){
    		rom = r;
    	}
    	
		String sk = context.getProperty("be.iminds.iot.dianne.rl.ale.skip");
		if (sk != null)
			this.skip = Integer.parseInt(sk);
		
		String ol = context.getProperty("be.iminds.iot.dianne.rl.ale.observationLength");
		if (ol != null)
			this.observationLength = Integer.parseInt(ol);
    	
    	// check if file exists
    	File f = new File(rom);
    	if(!f.exists()){
    		System.err.println("Failed to initialize ALE - ROM "+rom+" does not exist!");
    		throw new Exception("ROM "+rom+" does not exist!");
    	}
    	
    	loadROM(rom);
    	setFrameskip(skip);
    	
    	System.out.println("Loaded rom "+rom+", this has "+getActions()+" valid actions");
    	
    	reset();
    }
    
	@Override
	public float performAction(Tensor action) {
		int r = 0;
		
		if(observationLength > 1){
			float[] total = new float[observationLength*3*210*160];
			for(int i=0;i<observationLength;i++){
				r += performAction(factory.getTensorMath().argmax(action));
				System.arraycopy(getScreen(), 0, total, i*3*210*160, 3*210*160);
	    	}
			observation = factory.createTensor(total, observationLength, 3, 210, 160);
		} else {
			r = performAction(factory.getTensorMath().argmax(action));
	    	observation = factory.createTensor(getScreen(), 3, 210, 160);
		}
		
		
    	final float reward = r;
		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
    	
		return r > 0 ? 1 : r < 0 ? -1 : 0;
	}

	@Override
	public Tensor getObservation() {
		if(gameOver()){
			return null;
		} else {
			return observation;
		}
	}

	@Override
	public void reset() {
		resetGame();
		if(observationLength > 1 ){
	    	float[] total = new float[observationLength*3*210*160];
	    	for(int i=0;i<observationLength;i++){
	    		System.arraycopy(getScreen(), 0, total, i*3*210*160, 3*210*160);
	    	}
        	observation = factory.createTensor(total, observationLength, 3, 210, 160);
    	} else {
        	observation = factory.createTensor(getScreen(), 3, 210, 160);
    	}
	}

	private native void loadROM(String rom);

	private native int getActions();
	
	private native int performAction(int index);
	
	private native void resetGame();
	
	private native boolean gameOver();
	
	private native float[] getScreen();
	
	private native void setFrameskip(int skip);
	
	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
	
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
}
