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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(immediate=true,
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	service={ExperiencePool.class, Dataset.class},
	property={"aiolos.unique=true",
			  "aiolos.combine=*"})
public class FileExperiencePool implements ExperiencePool {

	private String name; // name of the repo
	private String dir; // directory where to store the samples
	private String[] labels; // labels for the actions
	private int stateSize; // size of the state Tensor
	private int actionSize; // size of the action Tensor
	private int maxSize = 1000000; // max size of the experience pool
	private boolean memoryOnly = false; // flag to store everything only in memory and not write to file
	
	private ReadWriteLock rwLock = new ReentrantReadWriteLock(true);
	
	// cycling arraylist of samples
	private ArrayList<ExperiencePoolSample> samples;
	private int start = 0; // current start index
	private int size = 0;
	
	private File file;
	private DataInputStream input;
	private DataOutputStream output;
	
	@Activate
	public void activate(Map<String, Object> config){
		this.name = (String) config.get("name");
		this.dir = (String) config.get("dir");
		this.labels = ((String) config.get("labels")).split(",");
		this.actionSize = Integer.parseInt((String) config.get("actionSize"));
		this.stateSize = Integer.parseInt((String) config.get("stateSize"));
		
		if(config.containsKey("maxSize"))
			this.maxSize = Integer.parseInt((String) config.get("maxSize"));
		
		if(config.containsKey("memoryOnly")){
			this.memoryOnly = Boolean.parseBoolean((String) config.get("memoryOnly"));
		}

		samples = new ArrayList<ExperiencePoolSample>(maxSize);
		
		// read from file 
		try {
			file = new File(dir+File.separator+"data.bin");
			
			if(!file.exists())
				file.createNewFile();
			
			input = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file, true)));

			// TODO do this more efficient in background thread?
			// TODO start reading from end of file (RandomAccessFile)?
			rwLock.writeLock().lock();
			try {
				while(true){
					float[] stateData = new float[stateSize];
					float[] actionData = new float[actionSize];
					float reward;
					float[] nextStateData = new float[stateSize];
					
					for(int i=0;i<stateSize;i++){
						stateData[i] = input.readFloat();
					}
					
					for(int i=0;i<actionSize;i++){
						actionData[i] = input.readFloat();
					}
					
					reward = input.readFloat();
					
					boolean terminate = false;
					for(int i=0;i<stateSize;i++){
						nextStateData[i] = input.readFloat();
						if(nextStateData[i]==Float.MIN_VALUE){
							terminate = true;
							break;
						}
					}
					
					Tensor state = new Tensor(stateData, stateSize);
					Tensor action = new Tensor(actionData, actionSize);
					Tensor nextState = terminate ? null : new Tensor(nextStateData, stateSize);
					
					ExperiencePoolSample s = new ExperiencePoolSample(state, action, reward, nextState);
					add(s);
				}
			} catch(IOException e){
				// read till exception end
			} finally {
				rwLock.writeLock().unlock();
			}
			
		} catch(IOException e){
			System.err.println("Failed to read experience pool file for "+name);
		}
	}
	
	@Override
	public int size() {
		return size;
	}

	@Override
	public Tensor getInputSample(Tensor t, int index) {
		t = getSample(index).state.copyInto(t);
		return t;
	}

	@Override
	public Tensor getState(Tensor t, int index) {
		t =  getSample(index).state.copyInto(t);
		return t;
	}

	@Override
	public Tensor getOutputSample(Tensor t, int index) {
		t =  getSample(index).action.copyInto(t);
		return t;
	}

	@Override
	public Tensor getAction(Tensor t, int index) {
		t =  getSample(index).action.copyInto(t);
		return t;
	}

	@Override
	public float getReward(int index) {
		return getSample(index).reward;
	}

	@Override
	public Tensor getNextState(Tensor t, int index) {
		t =  getSample(index).state.copyInto(t);
		return t;
	}
	
	public ExperiencePoolSample getSample(int index) {
		ExperiencePoolSample sample = null;
		try {
			rwLock.readLock().lock();
			sample = samples.get(getIndex(index));
		} finally {
			rwLock.readLock().unlock();
		}
		return sample;
	}
	
	public ExperiencePoolSample getSample(ExperiencePoolSample s, int index){
		ExperiencePoolSample ss = getSample(index);
		ss.action.copyInto(s.action);
		ss.state.copyInto(s.state);
		ss.nextState.copyInto(s.nextState);
		s.reward = ss.reward;
		return s;
	}
	
	private int getIndex(int index){
		int i = start + index;
		if(i >= size){
			i -= size;
		}
		return i;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public String[] getLabels() {
		return labels;
	}

	@Override
	public void addSample(Tensor state, Tensor action, float reward,
			Tensor nextState) {
		ExperiencePoolSample s = new ExperiencePoolSample(state, action, reward, nextState);
		add(s);
		
		if(!memoryOnly){
			write(s);
		}
	}
	
	@Override
	public void addSamples(Collection<ExperiencePoolSample> ss) {
		rwLock.writeLock().lock();
		for(ExperiencePoolSample s : ss){
			add(s);
		}
		rwLock.writeLock().unlock();
		
		if(!memoryOnly){
			synchronized(file){
				for(ExperiencePoolSample s : ss){
					write(s);
				}
			}
		}
	}
	
	private void add(ExperiencePoolSample s){
		rwLock.writeLock().lock();
		if(size==maxSize){
			// cycling
			int index = start++;
			if(start==maxSize){
				start = 0;
			}
			samples.set(index, s);
		} else {
			// just add to back
			size++;
			samples.add(s);
		}
		rwLock.writeLock().unlock();
	}
	
	private void write(ExperiencePoolSample s){
		// append to file
		synchronized(file){
			try {
				float[] stateData = s.state.get();
				for(int i=0;i<stateData.length;i++){
					output.writeFloat(stateData[i]);
				}
				
				float[] actionData = s.action.get();
				for(int i=0;i<actionData.length;i++){
					output.writeFloat(actionData[i]);
				}
				
				output.writeFloat(s.reward);
				
				if(s.nextState!=null){
					float[] nextStateData = s.nextState.get();
					for(int i=0;i<nextStateData.length;i++){
						output.writeFloat(nextStateData[i]);
					}
				} else {
					output.writeFloat(Float.MIN_VALUE);
				}
				
				output.flush();
			} catch(IOException e){
				System.err.println("Failed to write sample to file for experience pool "+name);
			}
		}
	}
	
	@Override
	public void reset() {
		rwLock.writeLock().lock();
		
		try {
			samples.clear();
			size = 0;
			start = 0;

			file.delete();
			file = new File(dir+File.separator+"data.bin");
		} finally {
			rwLock.writeLock().unlock();
		}
	}

	@Override
	public void setMaxSize(int max) {
		rwLock.writeLock().lock();

		// copy the samples
		ArrayList<ExperiencePoolSample> newSamples = new ArrayList<ExperiencePoolSample>(maxSize);
		
		if(size>0){
			int s = size - max;
			s = s > 0 ? s : 0;
			for(int i = s ; i< size; i++){
				newSamples.add(getSample(i));
			}
		}
		
		start = 0;
		size = size < max ? size : max;
		samples = newSamples;
		
		maxSize = max;
		
		rwLock.writeLock().unlock();
	}

	@Override
	public int[] inputDims() {
		return samples.get(0).input.dims();
	}

	@Override
	public int[] outputDims() {
		return samples.get(0).action.dims();
	}

}
