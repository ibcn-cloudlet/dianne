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
package be.iminds.iot.dianne.rnn.learn.sampling;

import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory.SamplingConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.sampling.config.SequenceSamplerConfig;

/**
 * Utility class that samples next sequence in separate thread
 * 
 * @author tverbele
 *
 */
public class SequenceSampler {

	private final SequenceDataset dataset;
	private final SequenceSamplingStrategy sampling;
	private final SequenceSamplerConfig config;
	
	private Sequence<Batch> sequenceInUse = null;
	private Sequence<Batch> sequenceBuffer = null;
	private volatile boolean ready = false;
	
	private Executor fetcher = Executors.newSingleThreadExecutor();
	
	public SequenceSampler(SequenceDataset d, SamplingConfig samplingStrategy, Map<String, String> config){
		this.dataset = d;
		this.sampling = SequenceSamplingFactory.createSamplingStrategy(samplingStrategy, d, config);;
		this.config = DianneConfigHandler.getConfig(config, SequenceSamplerConfig.class);
		
		// already fetch first batch
		fetchSequence();
	}
	
	/**
	 * Get next sequence from the dataset. Once you call this method, the previous 
	 * batch returned becomes obsolete and can be filled in with new data!
	 */
	public Sequence<Batch> nextSequence(){
		synchronized(this){
			if(!ready){
				try {
					this.wait();
				} catch (InterruptedException e) {
					throw new RuntimeException("Interrupted while fetching batch?!", e);
				}
			}
			
			ready = false;
		}
		
		// flip buffer and used sequence
		Sequence<Batch> temp = sequenceInUse;
		sequenceInUse = sequenceBuffer;
		sequenceBuffer = temp;
		
		// fetch new batch in buffer
		fetchSequence();
		
		return sequenceInUse;
	}
	
	protected void fetchSequence(){
		fetcher.execute(()->{
			int[] s = sampling.sequence(config.batchSize);
			int[] index = sampling.next(s, config.sequenceLength);
			sequenceBuffer = dataset.getBatchedSequence(sequenceBuffer, s, index, config.sequenceLength);
			ready=true;
			synchronized(SequenceSampler.this){
				SequenceSampler.this.notify();
			}
		});
	}
}
