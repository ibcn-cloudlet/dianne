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
package be.iminds.iot.dianne.nn.learn.criterion;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

public class CriterionFactory {
	
	public static enum CriterionConfig {
		MSE,
		NLL,
		ABS,
		BCE,
		GKL,
		HUB,
		GAU
	}
	
	public static class BatchConfig {
		
		public int batchSize = 1;
		
		public boolean batchAverage = true;
		
	}
	
	public static Criterion createCriterion(CriterionConfig c, int batchSize, boolean batchAverage){
		BatchConfig b = new BatchConfig();
		b.batchSize = batchSize;
		b.batchAverage = batchAverage;
		return createCriterion(c, b);
	}
	
	public static Criterion createCriterion(CriterionConfig c, Map<String, String> config){
		BatchConfig b = DianneConfigHandler.getConfig(config, BatchConfig.class);
		return createCriterion(c, b);
	}
	
	public static Criterion createCriterion(CriterionConfig c, BatchConfig b){
		Criterion criterion = null;
		
		switch(c) {
		case ABS :
			criterion = new AbsCriterion(b);
			break;
		case NLL :
			criterion = new NLLCriterion(b);
			break;
		case BCE :
			criterion = new BCECriterion(b);
			break;
		case GKL :
			criterion = new GaussianKLDivCriterion(b);
			break;
		case HUB :
			criterion = new PseudoHuberCriterion(b);
			break;
		case GAU :
			criterion = new GaussianCriterion(b);
			break;
		default:
			criterion = new MSECriterion(b);
			break;
		}
		
		return criterion;
	}
}
