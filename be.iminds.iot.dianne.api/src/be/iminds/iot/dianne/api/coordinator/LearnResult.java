package be.iminds.iot.dianne.api.coordinator;

/**
 * Summarizes the learn results
 * @author tverbele
 *
 */
public class LearnResult {

	/** avg error as perceived by the learner **/
	public float error;
	/** iterations executed **/
	public long iterations;
	
	
	public LearnResult(float error, long iterations){
		this.error = error;
		this.iterations = iterations;
	}
}
