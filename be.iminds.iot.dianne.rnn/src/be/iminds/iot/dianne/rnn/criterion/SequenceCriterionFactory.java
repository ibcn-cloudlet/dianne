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
package be.iminds.iot.dianne.rnn.criterion;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.CriterionConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

public class SequenceCriterionFactory {
	
	public static class SequenceCriterionConfig {
		
		/**
		 * Backpropagate the error at each step in the sequence or only for the last sample
		 */
		public boolean backpropAll = false;
		
	}
	
	public static SequenceCriterion createCriterion(CriterionConfig c, Map<String, String> config){
		Criterion criterion = CriterionFactory.createCriterion(c, config);
		SequenceCriterionConfig conf = DianneConfigHandler.getConfig(config, SequenceCriterionConfig.class);
		return new SequenceCriterion(criterion, conf);
	}
	
}
