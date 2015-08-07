package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

/**
 * The OutputManager provides an API to send output from neural network
 * Output modules to real things (devices) that can actuate upon this.
 * 
 * @author tverbele
 *
 */
public interface OutputManager {
	
	/**
	 * List the available things that can act as output
	 * 
	 * @return Descriptions of the available outputs
	 */
	List<OutputDescription> getAvailableOutputs();
	
	/**
	 * Connect the output of an Output module to an output thing.
	 * 
	 * @param outputId module ID of the neural network Output module
	 * @param nnId ID of the neural network instance
	 * @param output the real output to connect this Output module to
	 */
	void setOutput(UUID outputId, UUID nnId, String output);
	
	/**
	 * Disconnect the output of an Output module from an output thing.
	 * 
	 * @param outputId module ID of the neural network Output module
	 * @param nnId ID of the neural network instance
	 * @param output the real output to disconnect this Output module from
	 */
	void unsetOutput(UUID outputId, UUID nnId, String output);
}
