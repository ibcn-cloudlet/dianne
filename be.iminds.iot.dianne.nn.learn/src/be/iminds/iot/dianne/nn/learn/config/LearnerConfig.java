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
package be.iminds.iot.dianne.nn.learn.config;

import java.util.UUID;

public class LearnerConfig {

	/**
	 * The tag under which to publish the trained parameters
	 */
	public String tag;
	
	/**
	 * An optional tag to initialize the training from
	 */
	public String initTag;
	
	/**
	 * Start the training with new randomized parameters
	 */
	public boolean clean = false;
	
	/**
	 * Output intermediate results to the console
	 */
	public boolean trace = false;
	
	/**
	 * Interval to publish intermediate results
	 */
	public int traceInterval = 1000;
	
	/**
	 * Retry NaNretry times with last sync-ed parameters in case of a NaN loss
	 */
	public int NaNretry = 0;
	
	/**
	 * Learning strategy
	 */
	public String strategy = "FeedForwardLearningStrategy";
	
	/**
	 * Sync delta parameters with repository each syncInterval batches
	 */
	public int[] syncInterval = new int[]{1000};
	
	/**
	 * Interval to store nn parameters
	 */
	public int[] storeInterval = new int[]{0};
	
	/**
	 * module UUIDs to fix in this learn job - these modules won't get parameter updates
	 */
	public UUID[] fixed = new UUID[]{};
}
