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
package be.iminds.iot.dianne.rl.experience;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractExperiencePool extends AbstractDataset implements ExperiencePool {

	protected int maxSize = 1000000; // max number of samples in the experience pool
	
	protected int[] stateDims;
	protected int stateSize;
	protected int[] actionDims;
	protected int actionSize;
	protected int sampleSize;
	
	protected List<Integer> sequenceStarts = new ArrayList<>(); // start index of sequence
	protected List<Integer> sequenceLengths = new ArrayList<>(); // number of xp pool samples in sequence
	
	protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
	
	@Override
	public void activate(Map<String, Object> config) {
		super.activate(config);
		
		stateDims = inputDims;
		stateSize = inputSize;
	
		actionDims = targetDims;
		actionSize = targetSize;
	
		sampleSize = stateSize+actionSize+2;

		setup(config);
	}
	
	@Override
	protected void init(Map<String, Object> properties) {
		if(properties.containsKey("maxSize"))
			this.maxSize = Integer.parseInt((String) properties.get("maxSize"));
		
		String[] id = (String[])properties.get("stateDims");
		if(id!=null){
			inputDims= new int[id.length];
			for(int i=0;i<id.length;i++){
				inputDims[i] = Integer.parseInt(id[i]);
			}
		}
		
		String[] od = (String[])properties.get("actionDims");
		if(od!=null){
			targetDims= new int[od.length];
			for(int i=0;i<od.length;i++){
				targetDims[i] = Integer.parseInt(od[i]);
			}
		}
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
	public int size() {
		try {
			lock.readLock().lock();
			return noSamples;
		} finally {
			lock.readLock().unlock();
		}
	}
	
	@Override
	public int sequences(){
		try {
			lock.readLock().lock();
			return sequenceStarts.size();
		} finally {
			lock.readLock().unlock();
		}
	}
	
	@Override
	protected void readLabels(String labelsFile) {}
	
	@Override
	protected Tensor getInputSample(Tensor t, int index) {
		ExperiencePoolSample s = getSample(null, index);
		return s.input.copyInto(t);
	}

	@Override
	protected Tensor getTargetSample(Tensor t, int index) {
		ExperiencePoolSample s = getSample(null, index);
		return s.target.copyInto(t);
	}
	
	@Override
	public ExperiencePoolSample getSample(ExperiencePoolSample s, int index){
		try {
			lock.readLock().lock();
			s = getSample(s, index, true);
		} finally {
			lock.readLock().unlock();
		}
		return s;
	}
		
	@Override
	public ExperiencePoolBatch getBatch(ExperiencePoolBatch b, int... indices) {
		if(b == null){
			b = new ExperiencePoolBatch(indices.length, stateDims, actionDims);
		}
		
		try {
			lock.readLock().lock();
			int i = 0;
			for(int index : indices){
				getSample(b.getSample(i++), index, true);
			}
		} finally {
			lock.readLock().unlock();
		}
		return b;
	}

	
	@Override
	public List<ExperiencePoolSample> getSequence(List<ExperiencePoolSample> s, int sequence, int index, int length){
		if(s == null){
			s = new ArrayList<ExperiencePoolSample>(length == -1 ? 0 : length);
		}
		
		if(sequence > sequenceLengths.size()){
			throw new RuntimeException("Invalid sequence number");
		}
		
		try {
			lock.readLock().lock();
			
			int start = sequenceStarts.get(sequence);
			int l = sequenceLengths.get(sequence);
			
			if(index >= l){
				throw new RuntimeException("Invalid start index: "+index);
			}
			
			if(length == -1){
				length = l;
			}
			
			ExperiencePoolSample previous = null;
			for(int i=0;i<length;i++){
				ExperiencePoolSample sample = null;
				if(s.size() <= i){
					sample = new ExperiencePoolSample(previous == null ? null : previous.nextState, null, 0, null);
					s.add(sample);
				} else {
					sample = s.get(i);
					if(previous != null){
						sample.input = previous.nextState;
					}
				}
				
				if(i==0){
					getSample(sample, start, true);
				} else {
					getSample(sample, start+i, false);
				}
				
				previous = sample;
			}
			
		} finally {
			lock.readLock().unlock();
		}

		return s;
	}
	
	@Override
	public void addSequence(List<ExperiencePoolSample> sequence){
		if(sequence == null || sequence.isEmpty()){
			throw new RuntimeException("Null or empty sequence given");
		}

		int size = sequence.size();
		if(size > maxSize){
			// this cannot be stored in this pool
			System.out.println("Warning, a sequence of length "+size+" cannot be stored in this pool");
			return;
		}
		
		float[] buffer = new float[sampleSize];
		try {
			lock.writeLock().lock();

			int index = getBufferEnd();
			int start = index == maxSize ? 0 : index;
			
			for(ExperiencePoolSample s : sequence){
				System.arraycopy(s.input.get(), 0, buffer, 0 , stateSize);
				System.arraycopy(s.target.get(), 0, buffer, stateSize, actionSize);
				buffer[stateSize+actionSize] = s.getScalarReward();
				buffer[stateSize+actionSize+1] = s.isTerminal() ? 0.0f : 1.0f;
				
				if(index == maxSize){
					// cycle 
					index = 0;
					
					if(sequenceStarts.get(0) == 0){
						int removed = sequenceLengths.remove(0);
						noSamples-=removed;
						sequenceStarts.remove(0);
					}
				}

				if(getBufferStart() > 0 && index == getBufferStart()){
					int removed = sequenceLengths.remove(0);
					noSamples-=removed;
					sequenceStarts.remove(0);
				}
				
				writeData(index*sampleSize, buffer);
				index++;
			}
		
			sequenceStarts.add(start);
			sequenceLengths.add(size);
			noSamples+= size;
			
		} catch(Throwable t){ 
			t.printStackTrace();
		} finally {
			lock.writeLock().unlock();
		}
	}
	
	@Override
	public void reset() {
		try {
			noSamples = 0;
			sequenceLengths.clear();
			sequenceStarts.clear();
			lock.writeLock().lock();
		} finally {
			lock.writeLock().unlock();
		}
		
	}

	private ExperiencePoolSample getSample(ExperiencePoolSample s, int index, boolean loadState){
		float[] sampleBuffer = new float[sampleSize];
		
		loadData(getBufferPosition(index), sampleBuffer);
		
		if(s == null){
			s = new ExperiencePoolSample();	
		}
		
		// state
		// in case of sequences on might just reuse nextState tensor of previous instead of loading state again
		if(loadState){
			float[] stateBuffer = new float[stateSize];
			System.arraycopy(sampleBuffer, 0, stateBuffer, 0, stateSize);

			if(s.input == null){
				s.input = new Tensor(stateBuffer, stateDims);
			} else {
				s.input.set(stateBuffer);
			}
		}
		
		// action
		float[] actionBuffer = new float[actionSize];
		System.arraycopy(sampleBuffer, stateSize, actionBuffer, 0, actionSize);
		
		if(s.target == null){
			s.target = new Tensor(actionBuffer, actionDims);
		} else {
			s.target.set(actionBuffer);
		}
		
		// reward
		if(s.reward == null){
			s.reward = new Tensor(1);
		}
		s.reward.set(sampleBuffer[stateSize+actionSize], 0);
		
		// terminal
		if(s.terminal == null){
			s.terminal = new Tensor(1);
		}
		s.terminal.set(sampleBuffer[stateSize+actionSize+1], 0);
		
		if(!s.isTerminal()){
			// load next state
			float[] nextStateBuffer = new float[stateSize];
			loadData(getBufferPosition(index+1), nextStateBuffer);

			if(s.nextState == null){
				s.nextState = new Tensor(nextStateBuffer, stateDims);
			} else {
				s.nextState.set(nextStateBuffer);
			}
		} else {
			if(s.nextState == null){
				s.nextState = new Tensor(stateDims);
			} 
			s.nextState.fill(0);
		}
		
		return s;
	}
	
	private int getBufferPosition(int index){
		int startIndex = sequenceStarts.get(0);
		int pos = (startIndex+index) % maxSize;
		return pos*sampleSize;
	}
	
	private int getBufferStart(){
		if(sequenceStarts.isEmpty())
			return 0;
		return sequenceStarts.get(0);
	}
	
	private int getBufferEnd(){
		if(sequenceLengths.isEmpty())
			return 0;
		int last = sequenceLengths.size() - 1;
		return  (sequenceStarts.get(last)+sequenceLengths.get(last)) % maxSize;
	}
	
	protected abstract void setup(Map<String, Object> config);
	
	protected abstract void loadData(int position, float[] data);
	
	protected abstract void writeData(int position, float[] data);

}
