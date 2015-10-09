package be.iminds.iot.dianne.api.nn.learn;

/**
 * Represents the progress made by a Learner
 * 
 * @author tverbele
 *
 */
public class LearnProgress {

	/** The number of iterations (=number of batches) processed */
	public final long iteration;
	
	/** The current (avg) error perceived by the Learner */
	public final float error;
	
	public LearnProgress(long iteration, float error){
		this.iteration = iteration;
		this.error = error;
	}
}
