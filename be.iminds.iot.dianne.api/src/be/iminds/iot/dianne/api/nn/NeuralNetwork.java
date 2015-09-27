package be.iminds.iot.dianne.api.nn;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This is a convenient API for interacting with a Neural Network
 * 
 * @author tverbele
 *
 */
public interface NeuralNetwork {

	/**
	 * Get the Neural Network Instance DTO 
	 */
	NeuralNetworkInstanceDTO getNeuralNetworkInstance();
	
	/**
	 * Do a forward through the neural network synchronously, blocks until you get the output
	 * 
	 * Only works for neural network with single input, single output; else use the method with
	 * UUID inputId and UUID outputId args
	 */
	Tensor forward(Tensor input, String... tags);

	/**
	 * Do a forward through the neural network synchronously, blocks until you get the output
	 * 
	 */
	Tensor forward(UUID inputId, UUID outputId, Tensor input, String... tags);
	
	/**
	 * Asynchronously forward an input through the neural network. If you provided a ForwardListener
	 * this will be called on output.
	 *
	 * Only works for neural network with single input, single output; else use the method with
	 * UUID inputId arg
	 */
	void forward(ForwardListener callback, Tensor input, String... tags);

	/**
	 * Asynchronously forward an input through the neural network. If you provided a ForwardListener
	 * this will be called on output.
	 * 
	 */
	void forward(ForwardListener callback, UUID inputId, Tensor input, String... tags);

	
	/**
	 * Do a backward through the neural network synchronously, blocks until you get the gradInput
	 * 
	 * Only works for neural network with single input, single output; else use the method with
	 * UUID inputId and UUID outputId args
	 */
	Tensor backward(Tensor gradOutput, String... tags);

	/**
	 * Do a backward through the neural network synchronously, blocks until you get the gradInput
	 */
	Tensor backward(UUID outputId, UUID inputId, Tensor gradOutput, String... tags);

	
	/**
	 * Asynchronously backward an gradOutput through the neural network. If you provided a BackwardListener
	 * this will be called on gradInput.
	 * 
	 * Only works for neural network with single input, single output; else use the method with
	 * UUID inputId and UUID outputId args
	 */
	void backward(BackwardListener callback, Tensor gradOutput, String... tags);

	/**
	 * Asynchronously backward an gradOutput through the neural network. If you provided a BackwardListener
	 * this will be called on gradInput.
	 */
	void backward(BackwardListener callback, UUID outputId, Tensor gradOutput, String... tags);

	/**
	 * Get the input module in case of only one input
	 * @return
	 */
	Input getInput();
	
	/**
	 * Get the Input modules for this neural network instance
	 * @return
	 */
	Map<UUID, Input> getInputs();
	
	/**
	 * Get the output module in case of only one output
	 * @return
	 */
	Output getOutput();
	
	/**
	 * Get the Output modules for this neural network instance
	 * @return
	 */
	Map<UUID, Output> getOutputs();
	
	/**
	 * Get the output labels for the neural network output.
	 * 
	 * Only works for neural network with single output module.
	 * @return
	 */
	String[] getOutputLabels();
	
	/**
	 * Get output labels for the Output module with outputId
	 * @return
	 */
	String[] getOutputLabels(UUID outputId);
	
	/**
	 * Get Trainable modules for this neural network instance
	 * @return
	 */
	Map<UUID, Trainable> getTrainables();
	
	/**
	 * Get Preprocessor modules for this neural network instance
	 * @return
	 */
	Map<UUID, Preprocessor> getPreprocessors();
	
	/**
	 * Get all modules for this neural network instance
	 * @return
	 */
	Map<UUID, Module> getModules();
	
	/**
	 * Set parameters for this neural network instance
	 * @param parameters
	 */
	void setParameters(Map<UUID, Tensor> parameters);
	
	/**
	 * Set parameters for a single neural network instance module
	 * @param moduleId
	 * @param parameters
	 */
	void setParameters(UUID moduleId, Tensor parameters);
	
	/**
	 * Get all parameters for all neural network modules
	 */
	Map<UUID, Tensor> getParameters();
	

	/**
	 * Reinitialize the parameters with random values
	 */
	void resetParameters();
	
	/**
	 * Store the current parameters into the repository
	 * @param tag
	 */
	void storeParameters(String... tag);

	/**
	 * Store the delta compared to previous in the repository
	 * @param tag
	 */
	void storeDeltaParameters(Map<UUID, Tensor> previous, String... tag);
	
	/**
	 * Load parameters from the repository into the network instance
	 * @param tag
	 * @return the parameters loaded from the repository
	 * @throws exection when no parameters found with the given tags
	 */
	Map<UUID, Tensor> loadParameters(String... tag) throws Exception;
}
