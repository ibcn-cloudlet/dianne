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
package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory.SamplingConfig;
import be.iminds.iot.dianne.nn.learn.sampling.config.BatchSamplerConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

/**
 * Utility class that samples next batch in separate thread
 * 
 * @author tverbele
 *
 */
public class BatchSampler {

	// share fetcher threads for all samplers?
	private static Executor fetchers = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
	
	private final Dataset dataset;
	private final SamplingStrategy sampling;
	private final BatchSamplerConfig config;
	
	private LinkedBlockingQueue<BatchFetcher> ready;
	
	public BatchSampler(Dataset d, SamplingConfig samplingStrategy, Map<String, String> config){
		this.dataset = d;
		this.sampling = SamplingFactory.createSamplingStrategy(samplingStrategy, d, config);;
		this.config = DianneConfigHandler.getConfig(config, BatchSamplerConfig.class);
		
		this.ready = new LinkedBlockingQueue<>(this.config.fetchThreads);
		for(int i=0;i<this.config.fetchThreads;i++) {
			BatchFetcher fetcher = new BatchFetcher();
			fetcher.fetchBatch();
		}
		
	}
		
	public Batch nextBatch(){
		BatchFetcher fetcher;
		try {
			fetcher = ready.take();
			Batch b = fetcher.nextBatch();
			return b;
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
		
	}
	
	private class BatchFetcher {
		
		private Batch batchInUse = null;
		private Batch batchBuffer = null;
		
		/**
		 * Get next batch from the dataset. Once you call this method, the previous 
		 * batch returned becomes obsolete and can be filled in with new data!
		 */
		public Batch nextBatch(){
			// flip buffer and used batch
			Batch temp = batchInUse;
			batchInUse = batchBuffer;
			batchBuffer = temp;
			
			// fetch new batch in buffer
			fetchBatch();
			
			return batchInUse;
		}
		
		protected void fetchBatch(){
			fetchers.execute(()->{
				batchBuffer = dataset.getBatch(batchBuffer, sampling.next(config.batchSize));
				try {
					ready.put(BatchFetcher.this);
				} catch (InterruptedException e) {
				}
			});
		}
		
	}
	
	
}
