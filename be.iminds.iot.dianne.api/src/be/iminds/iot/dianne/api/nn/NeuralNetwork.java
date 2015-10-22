package be.iminds.iot.dianne.api.nn;

import java.lang.reflect.InvocationTargetException;
import java.util.Map;
import java.util.UUID;

import org.osgi.util.promise.Promise;

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
	 * Forward input through the neural network
	 * @param inputId id of the Input module to forward the input to
	 * @param outputId id of the Output module to get the output from
	 * @param input the input tensor
	 * @param tags optional array of tags
	 * @return result
	 */
	Promise<NeuralNetworkResult> forward(UUID inputId, UUID outputId, Tensor input, String... tags);

	/**
	 * Blocking call that returns the output Tensor after forwarding input.
	 * 
	 * This method is applicable for neural networks with one input and one output module.
	 * 
	 * @param input
	 * @param tags
	 * @return
	 */
	default Tensor forward(Tensor input, String... tags){
		Tensor result = null;
		Promise<NeuralNetworkResult> p = forward(null, null, input, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error forwarding input", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error forwarding input", e);
		}
		return result;
	}

	/**
	 * Backward gradOutput through the neural network
	 * @param outputId id of the Output module to backpropagate the gradOutput 
	 * @param inputId id of the Input module to gather the gradInput
	 * @param gradOutput the gradOutput tensor
	 * @param tags optional array of tags
	 * @return
	 */
	Promise<NeuralNetworkResult> backward(UUID outputId, UUID inputId, Tensor gradOutput, String... tags);

	/**
	 * Blocking call that returns the gradInput Tensor after back propagating gradOutput.
	 * 
	 * This method is applicable for neural networks with one input and one output module.
	 * 
	 * @param input
	 * @param tags
	 * @return
	 */
	default Tensor backward(Tensor gradOutput, String... tags){
		Tensor result = null;
		Promise<NeuralNetworkResult> p = backward(null, null, gradOutput, tags);
		try {
			if(p.getFailure()!=null){
				throw new RuntimeException("Error back propagating gradOutput", p.getFailure());
			}
		
			result = p.getValue().tensor;
		} catch(InterruptedException|InvocationTargetException e){
			throw new RuntimeException("Error back propagating gradOutput", e);
		}
		return result;
	}
	
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
