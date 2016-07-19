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
package be.iminds.iot.dianne.api.nn.util;

/**
 * The StrategyFactory can be used to magically construct various Strategy interfaces
 * in DIANNE: LearningStrategy, EvaluatorStrategy and ActionStrategy. These can even
 * be injected at runtime by provinding the source code, allowing a flexible experimentation
 * model.
 * 
 * @author tverbele
 *
 */

public interface StrategyFactory<T> {

	/**
	 * Create a new strategy. This can be done either by providing the classname of strategies
	 * known inside the DIANNE bundles, by providing a fully qualified classname (including package)
	 * for strategies exported in other bundles, or by providing source code of a strategy implementation
	 * as string arg.
	 *  
	 * @param strategy
	 * @return an instance of the strategy class
	 */
	T create(String strategy);
	
}
