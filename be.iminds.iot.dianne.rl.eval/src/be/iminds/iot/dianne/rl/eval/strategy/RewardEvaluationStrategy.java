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
package be.iminds.iot.dianne.rl.eval.strategy;

import java.util.List;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.eval.EvaluationStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.eval.strategy.config.RewardEvaluationConfig;

/**
 * Calculates the average (discounted) reward of all sequences in an experience pool
 * 
 * @author tverbele
 *
 */
public class RewardEvaluationStrategy implements EvaluationStrategy {

	protected ExperiencePool pool;
	protected RewardEvaluationConfig config;
	
	protected float reward = 0;
	protected int count = 0;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset "+dataset+" is not an experience pool");
		
		this.pool = (ExperiencePool) dataset;

		this.config = DianneConfigHandler.getConfig(config, RewardEvaluationConfig.class);
		
		// make sure the evaluator uses sequence granularity
		config.put("granularity", "SEQUENCE");
	}

	@Override
	public EvaluationProgress processIteration(long i) throws Exception {
		List<ExperiencePoolSample> sequence = pool.getSequence((int)(i++));
		
		float discountedReward = 0;
		for(int k = sequence.size()-1; k >= 0; k--){
			discountedReward = this.config.discount * discountedReward + sequence.get(k).getScalarReward();
		}

		reward += discountedReward;
		count++;
		
		return new EvaluationProgress(i, pool.sequences(), discountedReward);
	}

	@Override
	public Evaluation getResult() {
		Evaluation e = new Evaluation();
		e.size = count;
		e.metric = reward / count;
		return e;
	}
	

}
