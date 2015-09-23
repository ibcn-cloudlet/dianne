package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

/**
 * The DianneInputs interface provides an API to couple real inputs from things 
 * (devices) to actual neural network Input modules.
 * 
 * @author tverbele
 *
 */
public interface DianneInputs {

	/**
	 * List the available things that can act as input
	 * 
	 * @return Descriptions of the available inputs
	 */
	List<InputDescription> getAvailableInputs();
	
	/**
	 * Connect a real input to an Input module
	 * 
	 * @param inputId module ID of the neural network Input module
	 * @param nnId ID of the neural network instance
	 * @param input the real input to connect to this Input
	 */
	void setInput(UUID inputId, UUID nnId, String input);
	
	/**
	 * Disconnect a real input from an Input module
	 * 
	 * @param inputId module ID of the neural network Input module
	 * @param nnId ID of the neural network instance
	 * @param input the real input to disconnect this Input from
	 */
	void unsetInput(UUID inputId, UUID nnId, String input);
	
}
