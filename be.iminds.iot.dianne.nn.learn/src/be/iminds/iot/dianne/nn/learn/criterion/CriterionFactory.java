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

import be.iminds.iot.dianne.api.nn.learn.Criterion;

public class CriterionFactory {
	
	public enum CriterionConfig {
		MSE,
		NLL,
		ABS
	}
	
	public static Criterion createCriterion(CriterionConfig c){
		Criterion criterion = null;
		
		switch(c) {
		case ABS :
			criterion = new AbsCriterion();
			break;
		case NLL :
			criterion = new NLLCriterion();
			break;
		default:
			criterion = new MSECriterion();
			break;
		}
		
		return criterion;
	}
}
