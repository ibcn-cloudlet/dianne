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
package be.iminds.iot.dianne.rl.experience.adapters;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DatasetDTO;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.BatchedExperiencePoolSequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSequence;

@Component(
		service={Dataset.class, ExperiencePool.class},	
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.MultiExperiencePoolAdapter")
public class MultiExperiencePoolAdapter implements ExperiencePool {

	protected List<ExperiencePool> pools = Collections.synchronizedList(new ArrayList<>());
	protected String name;
	protected Map<String, Object> properties;
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE,
			policy=ReferencePolicy.DYNAMIC)
	public void addDataset(ExperiencePool p){
		this.pools.add(p);
		// TODO check whether all pools have same dimensions?!
	}
	
	public void removeDataset(ExperiencePool p){
		this.pools.remove(p);
	}
	
	@Activate
	public void activate(Map<String, Object> properties) {
		this.properties = properties;
		this.name = (String)properties.get("name");
	}

	@Override
	public DatasetDTO getDTO(){
		DatasetDTO dto = pools.get(0).getDTO();
		
		dto.name = getName();
		dto.inputDims = inputDims();
		dto.inputType = inputType();
		dto.targetDims = targetDims();
		dto.targetType = targetType();
		dto.size = size();
		dto.labels = getLabels();
		
		properties.entrySet().forEach(e -> {
			if(e.getKey().contains("."))
				return;
			
			for(Field f : DatasetDTO.class.getFields()){
				if(f.getName().equals(e.getKey()))
					return;
			}
			dto.properties.put(e.getKey(), e.getValue().toString());
		});
		
		return dto;
	}
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims() {
		return pools.get(0).inputDims();
	}
	
	@Override
	public String inputType(){
		return pools.get(0).inputType();
	}

	@Override
	public int[] targetDims() {
		return pools.get(0).targetDims();
	}

	@Override
	public String targetType(){
		return pools.get(0).inputType();
	}
	
	@Override
	public String[] getLabels() {
		return pools.get(0).getLabels();
	}

	@Override
	public int size() {
		return pools.stream().mapToInt(p -> p.size()).sum();
	}
	
	@Override
	public int sequences(){
		return pools.stream().mapToInt(p -> p.sequences()).sum();
	}
	
	@Override
	public int sequenceLength(int sequence){
		for(ExperiencePool pool : pools){
			int s = pool.sequences();
			if(sequence >= s){
				sequence -= s;
			} else {
				return pool.sequenceLength(sequence);
			}
		}
		return 0;
	}

	@Override
	public Sample getSample(Sample s, int index) {
		for(ExperiencePool pool : pools){
			int size = pool.size();
			if(index >= size){
				index -= size;
			} else {
				s = pool.getSample(s, index);
				return s;
			}
		}
		return s;
	}
	
	@Override
	public ExperiencePoolSample getSample(ExperiencePoolSample s, int index){
		for(ExperiencePool pool : pools){
			int size = pool.size();
			if(index >= size){
				index -= size;
			} else {
				s = pool.getSample(s, index);
				return s;
			}
		}
		return s;
	}
		
	@Override
	public ExperiencePoolBatch getBatch(ExperiencePoolBatch b, int... indices) {
		if(b == null){
			b = new ExperiencePoolBatch(indices.length, stateDims(), actionDims());
		}
		
		int i = 0;
		for(int index : indices){
			getSample(b.getSample(i++), index);
		}
		
		return b;
	}
	
	@Override
	public ExperiencePoolSequence getSequence(ExperiencePoolSequence s, int sequence, int index, int length){
		for(ExperiencePool pool : pools){
			int size = pool.sequences();
			if(sequence >= size){
				sequence -= size;
			} else {
				s = pool.getSequence(s, sequence, index, length);
				return s;
			}
		}
		return s;
	}
	
	@Override
	public BatchedExperiencePoolSequence getBatchedSequence(BatchedExperiencePoolSequence b, int[] sequences, int[] indices,
			int length) {
		if(b == null){
			b = new BatchedExperiencePoolSequence();
		}
		List<ExperiencePoolBatch> list = b.data; 
		
		// TODO reuse memory from the intermediate sequences fetched?
		// or better approach: fill in batch per batch directly (requires transforming the indices)
		List<Sequence<ExperiencePoolSample>> seqs = new ArrayList<>();
		for(int k=0;k<sequences.length; k++){
			seqs.add(getSequence(sequences[k], indices[k], length));
		}
		
		int minLength = length;
		for(Sequence<ExperiencePoolSample> s : seqs){
			if(s.size < minLength){
				minLength = s.size;
			}
		}
		
		for(int i=0; i<minLength; i++){
			ExperiencePoolBatch batch;
			if(list.size() > i){
				batch = list.get(i);
			} else {
				batch = new ExperiencePoolBatch(sequences.length, stateDims(), actionDims());
				list.add(batch);
			}
			
			for(int k=0; k<seqs.size(); k++){
				seqs.get(k).get(i).copyInto(batch.getSample(k));
			}
		}
		
		b.size = minLength;
		
		return b;
	}
	
	@Override
	public void addSequence(Sequence<ExperiencePoolSample> sequence){
		// add to random pool?
		ExperiencePool pool = pools.get((int)(Math.random()*pools.size()));
		pool.addSequence(sequence);
	}
	
	@Override
	public void reset() {
		pools.forEach(p -> p.reset());
	}
	
	@Override
	public void dump() {
		pools.forEach(p -> p.dump());
	}
	
}
