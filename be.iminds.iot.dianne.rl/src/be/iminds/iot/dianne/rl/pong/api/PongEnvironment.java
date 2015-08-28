package be.iminds.iot.dianne.rl.pong.api;

import be.iminds.iot.dianne.api.rl.Environment;

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
