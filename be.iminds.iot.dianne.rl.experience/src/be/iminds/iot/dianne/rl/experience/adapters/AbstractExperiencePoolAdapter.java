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

import java.io.IOException;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;

public abstract class AbstractExperiencePoolAdapter implements ExperiencePool {

	protected ExperiencePool pool;
	protected String name;
	
	@Reference
	void setDataset(ExperiencePool p){
		this.pool = p;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");
		configure(properties);
	}

	protected abstract void adaptFetchedSample(ExperiencePoolSample s);
	
	protected abstract void adaptAddingSample(ExperiencePoolSample s);
	
	protected abstract void configure(Map<String, Object> properties);
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims() {
		return pool.inputDims();
	}
	
	@Override
	public String inputType(){
		return pool.inputType();
	}

	@Override
	public int[] targetDims() {
		return pool.targetDims();
	}

	@Override
	public String targetType(){
		return pool.inputType();
	}
	
	@Override
	public String[] getLabels() {
		return pool.getLabels();
	}

	@Override
	public int size() {
		return pool.size();
	}
	
	@Override
	public int sequences(){
		return pool.sequences();
	}
	
	@Override
	public int sequenceLength(int sequence){
		return pool.sequenceLength(sequence);
	}

	@Override
	public Sample getSample(Sample s, int index) {
		s = pool.getSample(s, index);
		adaptFetchedSample((ExperiencePoolSample)s);
		return s;
	}
	
	@Override
	public ExperiencePoolSample getSample(ExperiencePoolSample s, int index){
		s = pool.getSample(s, index);
		adaptFetchedSample(s);
		return s;
	}
		
	@Override
	public ExperiencePoolBatch getBatch(ExperiencePoolBatch b, int... indices) {
		b =  pool.getBatch(b, indices);
		for(int i=0;i<indices.length;i++){
			adaptFetchedSample(b.getSample(i));
		}
		return b;
	}
	
	@Override
	public Sequence<ExperiencePoolSample> getSequence(Sequence<ExperiencePoolSample> s, int sequence, int index, int length){
		s = pool.getSequence(s, sequence, index, length);
		for(ExperiencePoolSample sample : s){
			adaptFetchedSample(sample);
		}
		return s;
	}
	
	@Override
	public Sequence<ExperiencePoolBatch> getBatchedSequence(Sequence<ExperiencePoolBatch> b, int[] sequences, int[] indices,
			int length) {
		b = pool.getBatchedSequence(b, sequences, indices, length);
		for(int i=0; i< sequences.length;i++){
			for(int j=0;j<indices.length;j++){
				ExperiencePoolSample s = b.get(i).getSample(j);
				adaptFetchedSample(s);
			}
		}
		return b;
	}
	
	@Override
	public void addSequence(Sequence<ExperiencePoolSample> sequence){
		for(ExperiencePoolSample sample : sequence){
			adaptAddingSample(sample);
		}
		pool.addSequence(sequence);
	}
	
	@Override
	public void reset() {
		pool.reset();
	}
	
	@Override
	public void dump() throws IOException {
		pool.dump();
	}
	
}
