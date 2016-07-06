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
package be.iminds.iot.dianne.rl.environment.gym;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.gym.config.GymConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import jep.Jep;
import jep.NDArray;

/**
 * Environment implementation for accessing OpenAI gym environments. Uses the Jep C-Python interface.
 * @author tverbele
 *
 */
@Component(immediate = true,
property = { "name="+GymEnvironment.NAME, "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class GymEnvironment implements Environment {

	public static final String NAME = "Gym";

	private ExecutorService thread = Executors.newSingleThreadExecutor();
	
	private GymConfig config;
	private boolean active = false;
	
	private Jep jep;
	
	private Tensor observation = new Tensor(6);
	private boolean end;
	
	@Activate
	void activate() {
		observation.fill(0.0f);
		try {
			thread.submit(()->{
				try {
					jep = new Jep(false);
					jep.eval("import sys");
					jep.eval("sys.argv = []");
					jep.eval("sys.argv.append('OpenAI Gym Environment')");
					jep.eval("import gym");
					
					// TODO check if successfully initialized?!
					
					// list envs
					jep.eval("from gym import envs");
					jep.eval("envs.registry.all()");
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Deactivate
	void deactivate(){
		try {
			thread.submit(()->{
				try {
					jep.close();
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Override
	public float performAction(Tensor action) {
		// TODO embed this in performAction callable?
		tensorToGymAction(action);
		
		try {
			return thread.submit(()->{
				try {
					jep.eval("o = env.step(action)");
					
					List o = (List)jep.getValue("o");
					Object r = o.get(1);
					float reward = 0.0f;
					if(r.getClass().equals(Double.class)){
						reward = ((Double)r).floatValue();
					} else if(r.getClass().equals(Integer.class)){
						reward = ((Integer)r).floatValue();
					} else {
						System.out.println("Invalid reward class?! "+r.getClass().getName());
					}
				
					// TODO return raw reward and let RL Learner scale it?!
					// squash between -1..1?!
					reward = reward/100;
					
					end = (Boolean)o.get(2);

					if(!end){
						NDArray nextState = (NDArray)o.get(0);
						observation = gymArrayToTensor(observation, nextState);
					} 
					
					
					if(this.config.render)
						jep.eval("env.render()");
					
					// TODO also allow use rendered rgb_arrays as observations?
					//jep.eval("screen = env.render('rgb_array')");
					//NDArray s = (NDArray)jep.getValue("screen");
					
					return reward;
				} catch(Exception e){
					e.printStackTrace();
				}
				return 0.0f;
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
		return 0;
	}

	@Override
	public Tensor getObservation(Tensor t) {
		if(end){
			return null;
		}
		return observation.copyInto(t);
	}

	@Override
	public void reset() {
		try {
			thread.submit(()->{
				try {
					end = false;
					jep.eval("init = env.reset()");
					
					// put this into observation
					NDArray state = (NDArray)jep.getValue("init");
					observation = gymArrayToTensor(observation, state);
					
				} catch(Exception e){
					e.printStackTrace();
				}					
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public void setup(Map<String, String> config) {
		if(active)
			throw new RuntimeException("This Environment is already active");
		
		active = true;
		
		this.config = DianneConfigHandler.getConfig(config, GymConfig.class);

		try {
			thread.submit(()->{
				try {
					jep.eval("env = gym.make('"+this.config.env+"')");
					
					// TODO check whether env is succesfully created
					
					System.out.println("Observation space:");
					jep.eval("env.observation_space");
					
					System.out.println("Action space:");
					jep.eval("env.action_space");
					
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
		
		reset();
	}

	@Override
	public void cleanup() {
		active = false;
	}

	private Tensor gymArrayToTensor(Tensor res, NDArray array){
		int[] dims = array.getDimensions();
		
		// swap in case of 3d array representing image data
		boolean image = false;
		if(
			array.isUnsigned() &&
			dims.length == 3 &&
			array.getData().getClass().getComponentType().equals(byte.class)
		){
			image = true;
			int[] swapped = new int[3];
			swapped[0] = dims[2];
			swapped[1] = dims[0];
			swapped[2] = dims[1];
			dims = swapped;
		}
		
		if(res == null){
			res = new Tensor(dims);
		} else {
			res.reshape(dims);
		}
		
		int size = 1;
		for(int d : dims){
			size *= d;
		}
		
		float[] tensorData = new float[size];
		if(array.getData().getClass().getComponentType().equals(double.class)){
			double[] data = (double[])array.getData();
		
			for(int i=0;i<data.length;i++){
				tensorData[i] = (float)data[i];
			}
		} else if(array.getData().getClass().getComponentType().equals(byte.class)){ 
			byte[] data = (byte[])array.getData();
			if(image){
				// swap data to our [channel, height, width] dimensions
				int k = 0;
				for(int c=0;c<dims[0];c++){
					for(int j=0;j<dims[1];j++){
						for(int i=0;i<dims[2];i++){
							tensorData[k++] =  ((int)(0xFF & data[c+j*dims[0]*dims[2]+i*dims[0]]))/255.0f;
						}
					}
				}
			} else {
				boolean unsigned = array.isUnsigned();
				for(int i=0;i<data.length;i++){
					tensorData[i] = unsigned ? ((int)(0xFF & data[i]))/255.0f : (float)data[i];
				}
			}
			
		} else {
			// TODO which other component types to support?!
			throw new RuntimeException("We do not support type "+array.getData().getClass().getComponentType().getName());
		}
		
		res.set(tensorData);
			
		//System.out.println(observation);
		return res;
	}
	
	private void tensorToGymAction(Tensor action){
		// TODO convert action tensor to what gym wants in action
		try {
			thread.submit(()->{
				try {
					// random sample for testing
					//jep.eval("action = env.action_space.sample()");

					// TODO for now only discrete actions are supported
					int discreteAction = TensorOps.argmax(action);
					jep.set("action", discreteAction);
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(Exception e){
			e.printStackTrace();
		}
		
	}
}
