package be.iminds.iot.dianne.rl.pong.api;

/**
 * PongListeners are called whenever an action is performed. Used to notify the UI
 * @author tverbele
 *
 */
public interface PongListener {

	/**
	 * Update the current state
	 */
	public void update(float x, float y, float vx, float vy, float p, float o);
	
	/**
	 * Notify of score
	 * @param player -1 = agent / 1 = opponent (AI or human)
	 */
	public void score(int player);
}
