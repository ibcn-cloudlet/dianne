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
package be.iminds.iot.dianne.rl.environment.kuka.config;

public class ReacherConfig {

	public static final int POSITION = 0;
	public static final int VELOCITY = 1;
	public static final int TORQUE = 2;
	public int mode = POSITION;
	
	/**
	 * Scale factor for lowering or increasing the min/max outputs.
	 * Can be used for all control modes but is not recommended in position control. 
	 * If value equals 1 the real min/max values are used.
	 * This value should be positive.
	 */
	public float outputScaleFactor = 1.0f;
	
	/**
	 * Fix gripper open if true
	 */
	public boolean gripperFixed = false;
}
