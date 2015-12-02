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
package be.iminds.iot.dianne.api.log;

/**
 * The DataLogger provides an interface for log various values and will print them
 * out nicely and/or calculate a running average.
 * 
 * @author tverbele
 *
 */
public interface DataLogger {

	/**
	 * Log a new entry - keys.length should equals number of values!
	 * @param label label for this set of key-values
	 * @param keys key labels for the values
	 * @param values the values to log
	 */
	void log(String label, String[] keys, float... values);

	/**
	 * Set an other than default interval for log messages with this label
	 * @param label 
	 * @param interval
	 */
	void setInterval(String label, int interval);
	
	/**
	 * Set an other than default alpha value for the running average of values with this key
	 * @param key
	 * @param alpha
	 */
	void setAlpha(String key, float alpha);
}
