package be.iminds.iot.dianne.rl.pong.api;

/**
 * PongListeners are called whenever an action is performed. Used to notify the UI
 * @author tverbele
 *
 */
public interface PongListener {

	public void update(float x, float y, float vx, float vy, float p, float o);
	
}
