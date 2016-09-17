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
package be.iminds.iot.dianne.nn.eval.strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.EvaluationStrategy;
import be.iminds.iot.dianne.nn.eval.strategy.config.EvaluationStrategyConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractEvaluationStrategy implements EvaluationStrategy {
	
	protected EvaluationStrategyConfig config;
	
	protected Dataset dataset;
	protected NeuralNetwork nn;
	
	protected float error;
	protected long total;
	
	protected Batch batch;
	protected Sample sample;
	protected int[] indices;
	
	protected EvaluationProgress progress;
	protected long tForward = 0;
	
	protected List<Tensor> outputs;

	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		this.nn = nns[0];
		
		this.config = DianneConfigHandler.getConfig(config, EvaluationStrategyConfig.class);
		indices = new int[this.config.batchSize];
		total = dataset.size();
		
		if(this.config.includeOutputs)
			outputs = new ArrayList<Tensor>();
		
		init(config);
	}

	@Override
	public EvaluationProgress processIteration(long i) throws Exception {
		Tensor out;
		if(config.batchSize > 1){
			// execute in batch
			if(i % config.batchSize != 0)
				return progress;
			
			// for last batch, adjust to remaining size
			if(i+indices.length > total){
				indices = new int[(int)(total-i)];
				batch = null;
			}
			
			for(int b=0;b<indices.length;b++)
				indices[b] = (int)(i+b);
			
			batch = dataset.getBatch(batch, indices);
			
			long t = System.nanoTime();
			out = nn.forward(batch.input);
			tForward += System.nanoTime() - t;
			
			if(outputs!=null)
				for(int k=0;k<this.config.batchSize;k++)
					outputs.add(out.select(0, k).copyInto(null));
			
			float err = eval(out, batch.target);
			
			error += err;

		} else {
			sample = dataset.getSample(sample, (int)i);
			
			long t = System.nanoTime();
			out = nn.forward(sample.input);
			tForward += System.nanoTime() - t;
			
			if(outputs!=null)
				outputs.add(out.copyInto(null));

			float err = eval(out, sample.target);
			error += err;
		}

		return progress = new EvaluationProgress(i+config.batchSize, total, error/(i+config.batchSize));
	}

	@Override
	public Evaluation getResult() {
		Evaluation eval = finish();
		eval.total = total;
		eval.error = error/total;
		eval.forwardTime = (tForward/1000000f)/total;
		eval.outputs = outputs;
		return eval;
	}

	protected abstract float eval(Tensor output, Tensor target);
	
	protected abstract void init(Map<String, String> config);
	
	protected abstract Evaluation finish();
}

