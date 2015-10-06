package be.iminds.iot.dianne.api.nn.learn;

/**
 * Strategy to select a next index to visit in the learning procedure.
 * 
 * @author tverbele
 *
 */
public interface SamplingStrategy {

	/**
	 * @return next index of the dataset to visit
	 */
	int next();
}
