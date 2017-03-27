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
package be.iminds.iot.dianne.rl.learn.sampling;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory.SamplingConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.sampling.config.ExperienceSamplerConfig;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingStrategy;

/**
 * Utility class that samples next experience batch/sequence in separate thread
 * 
 * @author tverbele
 *
 */
public class ExperienceSampler {

	private final ExperiencePool pool;
	private final ExperienceSamplerConfig config;

	// select sampling strategy depending on sequence/batch mode
	private SamplingStrategy sampling;
	private SequenceSamplingStrategy ssampling;
	
	private ExperiencePoolBatch batchInUse = null;
	private ExperiencePoolBatch batchBuffer = null;
	private Sequence<ExperiencePoolBatch> sequenceInUse = null;
	private Sequence<ExperiencePoolBatch> sequenceBuffer = null;
	private volatile boolean ready = false;
	
	private Executor fetcher = Executors.newSingleThreadExecutor();
	
	public ExperienceSampler(ExperiencePool p, SamplingConfig samplingStrategy, Map<String, String> config){
		this.pool = p;
		this.config = DianneConfigHandler.getConfig(config, ExperienceSamplerConfig.class);

		if(this.config.sequenceLength > 1){
			this.ssampling = SequenceSamplingFactory.createSamplingStrategy(samplingStrategy, pool, config);;
		} else {
			this.sampling = SamplingFactory.createSamplingStrategy(samplingStrategy, pool, config);;
		}
	}
	
	public ExperiencePoolBatch nextBatch(){
		if(config.sequenceLength > 1){
			// just get first batch from sequence?!
			// one should not really be calling nextBatch if he/she wants sequences...
			return nextSequence().get(0);
		}
		
		synchronized(this){
			if(!ready){
				try {
					// first call - fetch
					if(batchBuffer == null && batchInUse == null)
						fetchBatch();
					
					this.wait();
				} catch (InterruptedException e) {
					throw new RuntimeException("Interrupted while fetching batch?!", e);
				}
			}
			
			ready = false;
		}
		
		// flip buffer and used batch
		ExperiencePoolBatch temp = batchInUse;
		batchInUse = batchBuffer;
		batchBuffer = temp;
		
		// fetch new batch in buffer
		fetchBatch();
		
		return batchInUse;
	}
	
	/**
	 * Get next sequence from the dataset. Once you call this method, the previous 
	 * batch returned becomes obsolete and can be filled in with new data!
	 */
	public Sequence<ExperiencePoolBatch> nextSequence(){
		if(config.sequenceLength <= 1){
			// one shouldn't be calling nextSequence when only 1 batch is requested
			return new Sequence<ExperiencePoolBatch>(Collections.singletonList(nextBatch()));
		}
		
		synchronized(this){
			if(!ready){
				try {
					// first call - fetch
					if(sequenceBuffer == null && sequenceInUse == null)
						fetchSequence();
					
					this.wait();
				} catch (InterruptedException e) {
					throw new RuntimeException("Interrupted while fetching batch?!", e);
				}
			}
			
			ready = false;
		}
		
		// flip buffer and used sequence
		Sequence<ExperiencePoolBatch> temp = sequenceInUse;
		sequenceInUse = sequenceBuffer;
		sequenceBuffer = temp;
		
		// fetch new batch in buffer
		fetchSequence();
		
		return sequenceInUse;
	}
	
	protected void fetchSequence(){
		fetcher.execute(()->{
			int[] s = ssampling.sequence(config.batchSize);
			int[] index = ssampling.next(s, config.sequenceLength);
			sequenceBuffer = pool.getBatchedSequence(sequenceBuffer, s, index, config.sequenceLength);
			ready=true;
			synchronized(ExperienceSampler.this){
				ExperienceSampler.this.notify();
			}
		});
	}
	
	protected void fetchBatch(){
		fetcher.execute(()->{
			batchBuffer = pool.getBatch(batchBuffer, sampling.next(config.batchSize));
			ready=true;
			synchronized(ExperienceSampler.this){
				ExperienceSampler.this.notify();
			}
		});
	}
}
