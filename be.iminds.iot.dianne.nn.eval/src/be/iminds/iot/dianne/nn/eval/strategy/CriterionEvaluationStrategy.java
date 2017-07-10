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

import java.util.Map;

import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class CriterionEvaluationStrategy extends AbstractEvaluationStrategy {
	
	private Criterion criterion;
	
	@Override
	protected void init(Map<String, String> config) {
		config.put("batchAverage", "false");
		criterion = CriterionFactory.createCriterion(super.config.criterion, config);
	}
	
	@Override
	protected void update(Map<String, String> config) {
		config.put("batchAverage", "false");
		criterion = CriterionFactory.createCriterion(super.config.criterion, config);
	}

	protected float eval(Tensor output, Tensor target){
		return TensorOps.sum(criterion.loss(output, target));
	}

	@Override
	protected Evaluation finish() {
		return new Evaluation();
	}

}

