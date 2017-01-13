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

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ExecutionException;
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
	
	private Tensor observation;
	private boolean end;
	
	private int[] actionDims;
	private boolean discrete = true;
	
	private int[] observationDims;
	
	private int count = 0;
	
	@Activate
	void activate() {
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
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
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
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
			e.printStackTrace();
		}
	}
	
	@Override
	public int[] observationDims() {
		return observationDims;
	}

	@Override
	public int[] actionDims() {
		return actionDims;
	}
		
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		// TODO embed this in performAction callable?
		action = TensorOps.mul(action, action, this.config.actionFactor);
		tensorToGymAction(action);
		
		try {
			return thread.submit(()->{
				try {
					jep.eval("observation, reward, done, info = env.step(action)");
					
					NDArray<?> nextState = (NDArray<?>)jep.getValue("observation");
					Object r = jep.getValue("reward");
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
					reward = reward * this.config.rewardFactor;

					end = (Boolean) jep.getValue("done");
					if (this.config.maxActions != -1 && count++ > this.config.maxActions) {
						end = true;
					} 
					if(!end){	
						observation = gymArrayToTensor(observation, nextState);
					} else {
						count = 0;
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
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
			e.printStackTrace();
		}
		return 0;
	}

	@Override
	public Tensor getObservation(Tensor t) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		if(end){
			return null;
		}
		return observation.copyInto(t);
	}

	@Override
	public void reset() {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		try {
			thread.submit(()->{
				try {
					count = 0;
					end = false;
					jep.eval("init = env.reset()");
					
					// put this into observation
					NDArray<?> state = (NDArray<?>)jep.getValue("init");
					observation = gymArrayToTensor(observation, state);
					
				} catch(Exception e){
					e.printStackTrace();
				}					
			}).get();
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
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
					jep.eval("os = env.observation_space");
					String o = (String)jep.getValue("os");
					System.out.println(o);
					
					// TODO We need to also incorporate more meta info, like the ranges etc?
					// Introduce similar concept as "space" for environments?
					if(o.startsWith("Box")){
						String ints = o.substring(4, o.length()-1);
						String[] split = ints.split(",");
						int i = 0;
						for(int k=0;k<split.length;k++){
							if(split[k].length() > 0)
								i++;
						}
						int[] dims = new int[i];
						i = 0;
						for(int k=0;k<split.length;k++){
							if(split[k].length() > 0)
								dims[i++] = Integer.parseInt(split[k].trim());
						}
						observationDims = dims;
						if(dims.length == 3){
							// swap
							observationDims = new int[dims.length];
							observationDims[0] = dims[2];
							observationDims[1] = dims[0];
							observationDims[2] = dims[1];
						}
					} else {
						System.out.println("Unknown observation space");
					}
					System.out.println(Arrays.toString(observationDims));
					observation = new Tensor(observationDims);

					
					System.out.println("Action space:");
					jep.eval("acs = env.action_space");
					String a = (String)jep.getValue("acs");
					System.out.println(a);
					
					if(a.startsWith("Discrete")) {
						String dim = a.substring(9, a.length()-1);
						actionDims = new int[]{Integer.parseInt(dim)};
					} else if(a.startsWith("High-Low")) {
						// TODO is this right?!
						discrete = false;
						String dim = a.substring(9, a.indexOf(","));
						actionDims = new int[]{Integer.parseInt(dim)};
					} else if(a.startsWith("Box")) {
						discrete = false;
						String dim = a.substring(4, a.indexOf(","));
						actionDims = new int[]{Integer.parseInt(dim)};
					} else {
						System.out.println("Unknown action space");
					}
					System.out.println(Arrays.toString(actionDims));
					
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
			e.printStackTrace();
		}
		
		reset();
	}

	@Override
	public void cleanup() {
		active = false;
	}

	private Tensor gymArrayToTensor(Tensor res, NDArray<?> array){
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
					// generate random sample for testing
					// jep.eval("action = env.action_space.sample()");
					
					
					if(discrete){
						int discreteAction = TensorOps.argmax(action);
						jep.set("action", discreteAction);
					} else {
						NDArray<float[]> a = new NDArray<float[]>(action.get());
						jep.set("action", a);
					}
				} catch(Exception e){
					e.printStackTrace();
				}
			}).get();
		} catch(InterruptedException e){
			// ignore interrupted exceptions
		} catch(ExecutionException e){
			e.printStackTrace();
		}
		
	}

}
