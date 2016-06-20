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
package be.iminds.iot.dianne.nn.eval;

import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.eval.config.CriterionEvaluatorConfig;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service={Evaluator.class},
		property={"aiolos.unique=true",
		"dianne.evaluator.category=CRITERION"})
public class CriterionEvaluator extends AbstractEvaluator {
	
	private Criterion criterion;
	
	@Override
	protected void init(Map<String, String> config) {
		CriterionEvaluatorConfig c = DianneConfigHandler.getConfig(config, CriterionEvaluatorConfig.class);
		criterion = CriterionFactory.createCriterion(c.criterion);
	}

	protected float evalOutput(int index, Tensor out, Tensor expected){
		
		Tensor error = criterion.error(out, expected);
		
		if(this.config.trace){
			System.out.println("Sample "+index+" error is "+error.get(0));
		}
		
		return error.get(0);
	}

	@Override
	protected Evaluation finish() {
		return new Evaluation();
	}

}

