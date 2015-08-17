package be.iminds.iot.dianne.api.nn.learn;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A Processor contains the logic to return the next update to gradParameters
 * during learning. 
 *   
 * @author tverbele
 *
 */
public interface Processor {
	
	/**
	 * Get the next parameters update
	 * @return a gradParameters tensor for each trainable module from the neural network, or null if no further processing is possible.
	 */
	Map<UUID, Tensor> processNext();
	
}
