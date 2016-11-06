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
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import be.iminds.iot.dianne.api.dataset.AbstractDataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractExperiencePool extends AbstractDataset implements ExperiencePool {

	protected long maxSize = 100000000; // max size of the experience pool (in bytes)
	
	protected int[] stateDims;
	protected int stateSize;
	protected int[] actionDims;
	protected int actionSize;
	protected int sampleSize;
	
	protected float[] emptyState;
	
	protected LinkedList<Integer> sequenceStarts = new LinkedList<>(); // start position in buffer
	protected LinkedList<Integer> sequenceLengths = new LinkedList<>(); // number of xp pool samples in sequence

	protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
	
	@Override
	public void activate(Map<String, Object> config) {
		super.activate(config);
		
		stateDims = inputDims;
		stateSize = inputSize;
	
		actionDims = targetDims;
		actionSize = targetSize;
	
		sampleSize = 2*stateSize+actionSize+1;

		emptyState = new float[stateSize];
		Arrays.fill(emptyState, Float.NaN);
		
		setup(config);
	}
	
	@Override
	protected void init(Map<String, Object> properties) {
		if(properties.containsKey("maxSize"))
			this.maxSize = Long.parseLong((String) properties.get("maxSize"));
		
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
	protected void readLabels(String labelsFile) {}
	
	// obsolete for ExperiencePool?
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
			
			int pos = getBufferPosition(index);
			s = getSample(s, pos, true);
		} finally {
			lock.readLock().unlock();
		}
		return s;
	}
	
	private ExperiencePoolSample getSample(ExperiencePoolSample s, int position, boolean loadState){
		float[] sampleBuffer = loadData(position);
		float[] actionBuffer = new float[actionSize];
		float[] nextStateBuffer = new float[stateSize];

		System.arraycopy(sampleBuffer, stateSize, actionBuffer, 0, actionSize);
		float rewardBuffer = sampleBuffer[stateSize+actionSize];
		System.arraycopy(sampleBuffer, stateSize+actionSize+1, nextStateBuffer, 0, stateSize);
		
		if(s == null){
			s = new ExperiencePoolSample();	
		}
		
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
		
		if(s.target == null){
			s.target = new Tensor(actionBuffer, actionDims);
		} else {
			s.target.set(actionBuffer);
		}
		
		s.reward = rewardBuffer;
		
		if(s.nextState == null){
			s.nextState = new Tensor(nextStateBuffer, stateDims);
		} else {
			s.nextState.set(nextStateBuffer);
		}
		
		s.isTerminal = Float.isNaN(nextStateBuffer[0]);
		
		return s;
	}
	
	private int getBufferPosition(int index){
		int k = 0;
		Iterator<Integer> lengths = sequenceLengths.iterator();
		Iterator<Integer> starts = sequenceStarts.iterator();

		int length = lengths.next();
		int start = starts.next();
		
		while(k + length <= index){
			k += length;
			
			length = lengths.next();
			start = starts.next();
		}
		int positionInSequence = index - k;
		
		return start + (stateSize+actionSize+1)*positionInSequence;
	}

	private int getBufferStart(){
		if(sequenceStarts.isEmpty())
			return 0;
		return sequenceStarts.getFirst();
	}
	
	private int getBufferEnd(){
		if(sequenceStarts.isEmpty())
			return 0;
		return sequenceStarts.getLast()+sequenceLengths.getLast()*(stateSize+actionSize+1)+stateSize;
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
			int offset = stateSize+actionSize+1;

			
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
					getSample(sample, start+i*offset, false);
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
		int offset = stateSize+actionSize+1;
		float[] buffer = new float[sequence.size()*offset+stateSize];
		int i = 0;
		for(ExperiencePoolSample s : sequence){
			System.arraycopy(s.input.get(), 0, buffer, offset*i , stateSize);
			System.arraycopy(s.target.get(), 0, buffer, offset*i+stateSize, actionSize);
			buffer[offset*i+stateSize+actionSize] = s.reward;
			i++;
		}
		ExperiencePoolSample last = sequence.get(size-1);
		System.arraycopy(last.isTerminal ? emptyState : last.nextState.get(), 0, buffer, offset*(i-1)+stateSize+actionSize+1, stateSize);

		int pos = 0, l = 0;
		try {
			lock.writeLock().lock();

			// write the buffer - check whether we have to cycle
			pos = getBufferEnd();
			
			l = buffer.length;
			if(l >= maxSize/4){
				// this cannot be stored in this pool
				System.out.println("Warning, a sequence of length "+size+" cannot be stored in this pool");
				return;
			}
			
			if(pos + l >= maxSize/4){
				// cycle
				pos = 0;
				
				int first = -1;
				while(first != 0){
					int removed = sequenceLengths.removeFirst();
					noSamples-=removed;
					first = sequenceStarts.removeFirst();
				}
			}
			
			while(getBufferStart() > 0 && pos + l > getBufferStart()){
				int removed = sequenceLengths.removeFirst();
				noSamples-=removed;
				sequenceStarts.removeFirst();
			}
			
			writeData(pos, buffer);

			sequenceStarts.addLast(pos);
			sequenceLengths.addLast(size);
			noSamples+= size;
			
		} catch(Throwable t){ 
			System.out.println("SHIT "+pos+" "+l+" "+maxSize/4+" "+getBufferStart()+" "+sequenceStarts.size());
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
	
	protected abstract void setup(Map<String, Object> config);
	
	protected abstract float[] loadData(int index);
	
	protected abstract void writeData(int index, float[] sample);

}
