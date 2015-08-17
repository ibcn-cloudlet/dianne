package be.iminds.iot.dianne.api.nn.learn;


/**
 * A Processor contains the logic to return the next update to gradParameters
 * during learning. 
 *   
 * @author tverbele
 *
 */
public interface Processor {
	
	/**
	 * Update gradParameters of all trainable modules.
	 * @return a measure of the error
	 */
	float processNext();
	
}
