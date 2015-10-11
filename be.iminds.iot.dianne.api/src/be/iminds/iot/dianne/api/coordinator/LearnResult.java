package be.iminds.iot.dianne.api.coordinator;

/**
 * Summarizes the learn results
 * @author tverbele
 *
 */
public class LearnResult {

	/** Accuracy on the test set **/
	public float accuracy;
	/** Time in ms to forward one sample **/
	public float forwardTime;
	
	
	public LearnResult(float accuracy, float forwardTime){
		this.accuracy = accuracy;
		this.forwardTime = forwardTime;
	}
}
