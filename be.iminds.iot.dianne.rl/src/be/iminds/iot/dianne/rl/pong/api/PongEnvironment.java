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
package be.iminds.iot.dianne.rl.pong.api;

import be.iminds.iot.dianne.api.rl.environment.Environment;

/**
 * PongEnvironment service for the Pong UI bundle
 * 
 * @author tverbele
 *
 */
public interface PongEnvironment extends Environment {

	
	/**
	 * Get the field bounds  
	 * @return bound between 0 and 1 
	 */
	float getBounds();
	
	/**
	 * Get the game speed (v_def)
	 * @return game speed
	 */
	float getSpeed();
	
	/**
	 * Get the paddle length
	 * @return paddle length
	 */
	float getPaddleLength();
	
	/**
	 * Get the width of the paddle
	 * @return paddle width
	 */
	float getPaddleWidth();
	
	/**
	 * Get the radius of the ball
	 * @return radius
	 */
	float getBallRadius();
	
	/**
	 * Turn on/of the AI
	 */
	void useAI(boolean ai);
	
	/**
	 * For manual control instead of AI
	 * @param action
	 */
	void setOpponentAction(int action);
}
