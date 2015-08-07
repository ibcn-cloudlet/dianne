package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Called when a Module has done a backward pass. 
 * 
 * This interface can be registered as an OSGi service with the service property
 *  targets = String[]{nnId:moduleId, nnId:moduleId2}
 * in order to listen to only specific modules from certain neural network instances.
 * 
 * In case only a nnId is given as targets property, this backward listener will only 
 * listen to the Input modules of this neural network instance.
 *   
 * @author tverbele
 *
 */
public interface BackwardListener {

	/**
	 * Called when a Module has performed a backward pass
	 * @param output a copy of the gradInput data
	 * @param tags a copy of the tags provided
	 */
	void onBackward(final Tensor gradInput, final String... tags);
}
