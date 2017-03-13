/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.api.nn;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Memory;
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
	 * Get the Neural Network Instance UUID
	 * @return
	 */
	UUID getId();
	
	/**
	 * Get the Neural Network Instance DTO 
	 */
	NeuralNetworkInstanceDTO getNeuralNetworkInstance();
	
	/**
	 * Get a UUID for a module by its name. Casing is ignored. A name is not necessary unique, in which case
	 * this method can return any one of the matching UUIDs.
	 */
	UUID getModuleId(String name);
	
	/**
	 * Get UUIDs for modules by their name. Casing is ignored. A name is not necessary unique, in which case
	 * this method can return any one of the matching UUIDs.
	 */
	UUID[] getModuleIds(String... name);
	
	/**
	 * Forward input through the neural network
	 * @param inputId id of the Input module to forward the input to
	 * @param outputId id of the Output module to get the output from - or null for all
	 * @param input the input tensor
	 * @param tags optional array of tags
	 * @return result
	 */
	Promise<NeuralNetworkResult> forward(UUID inputId, UUID outputId, Tensor input, String... tags);
	
	/**
	 * Forward inputs through the neural network
	 * @param inputIds ids of the Input modules to forward the inputs to
	 * @param outputIds ids of the Output modules to get the outputs from - or null for all
	 * @param inputs the input tensors
	 * @param tags optional array of tags
	 * @return result
	 */
	Promise<NeuralNetworkResult> forward(UUID[] inputId, UUID[] outputId, Tensor[] input, String... tags);

	/**
	 * Blocking call that returns the output Tensor after forwarding input.
	 * 
	 * This method is applicable for neural networks with one input and one output module.
	 * 
	 * @param input
	 * @param tags
	 * @return
	 */
	Tensor forward(Tensor input, String... tags);

	
	Promise<NeuralNetworkSequenceResult> forward(UUID inputId, UUID outputId, List<Tensor> input, String... tags);
	
	Promise<NeuralNetworkSequenceResult> forward(UUID[] inputId, UUID[] outputId, List<Tensor>[] input, String... tags);

	List<Tensor> forward(List<Tensor> input, String... tags);

	
	
	/**
	 * Backward gradOutputs through the neural network
	 * @param outputId id of the Output module to backpropagate the gradOutput 
	 * @param inputId id of the Input module to gather the gradInput - or null for all
	 * @param gradOutput the gradOutput tensor
	 * @param accGradParameters also accumulate gradient to the neural network's parameters
	 * @param tags optional array of tags
	 * @return
	 */
	Promise<NeuralNetworkResult> backward(UUID outputId, UUID inputId, Tensor gradOutput, boolean accGradParameters, String... tags);

	default Promise<NeuralNetworkResult> backward(UUID outputId, UUID inputId, Tensor gradOutput, String... tags){
		return backward(outputId, inputId, gradOutput, false, tags);
	}

	
	/**
	 * Backward gradOutput through the neural network
	 * @param outputIds ids of the Output modules to backpropagate the gradOutputs 
	 * @param inputIds ids of the Input modules to gather the gradInputs - or null for all
	 * @param gradOutputs the gradOutput tensors
	 * @param accGradParameters also accumulate gradient to the neural network's parameters
	 * @param tags optional array of tags
	 * @return
	 */
	Promise<NeuralNetworkResult> backward(UUID[] outputIds, UUID[] inputIds, Tensor[] gradOutputs, boolean accGradParameters, String... tags);

	default Promise<NeuralNetworkResult> backward(UUID[] outputIds, UUID[] inputIds, Tensor[] gradOutputs, String... tags){
		return backward(outputIds, inputIds, gradOutputs, false, tags);
	}

	
	/**
	 * Blocking call that returns the gradInput Tensor after back propagating gradOutput.
	 * 
	 * This method is applicable for neural networks with one input and one output module.
	 * 
	 * @param input
	 * @param accGradParameters also accumulate gradient to the neural network's parameters
	 * @param tags
	 * @return
	 */
	Tensor backward(Tensor gradOutput, boolean accGradParameters, String... tags);
	
	default Tensor backward(Tensor gradOutput, String... tags){
		return backward(gradOutput, false, tags);
	}
	
	
	
	Promise<NeuralNetworkSequenceResult> backward(UUID outputId, UUID inputId, List<Tensor> gradOutput, boolean accGradParameters, String... tags);
	
	Promise<NeuralNetworkSequenceResult> backward(UUID[] outputIds, UUID[] inputIds, List<Tensor>[] gradOutputs, boolean accGradParameters, String... tags);

	List<Tensor> backward(List<Tensor> gradOutput, boolean accGradParameters, String... tags);
	
	
	
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
	 * Get Input module by id
	 * @param inputId
	 * @return
	 */
	default Input getInput(UUID inputId){
		return getInputs().get(inputId);
	}
	
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
	 * Get Output module by id
	 * @param outputId
	 * @return
	 */
	default Output getOutput(UUID outputId){
		return getOutputs().get(outputId);
	}
	
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
	 * Set the output labels for the neural network output.
	 * 
	 * Only works for neural network with single output module.
	 * @return
	 */
	void setOutputLabels(String[] labels);
	
	/**
	 * Set output labels for the Output module with outputId
	 * @return
	 */
	void setOutputLabels(UUID outputId, String[] labels);
	
	
	/**
	 * Get Trainable modules for this neural network instance
	 * @return
	 */
	Map<UUID, Trainable> getTrainables();
	
	/**
	 * Get Trainable module by id
	 * @param moduleId
	 * @return
	 */
	default Trainable getTrainable(UUID moduleId){
		return getTrainables().get(moduleId);
	}
	
	/**
	 * Get Preprocessor modules for this neural network instance
	 * @return
	 */
	Map<UUID, Preprocessor> getPreprocessors();
	
	/**
	 * Get Preprocessor module by id
	 * @param moduleId
	 * @return
	 */
	default Preprocessor getPreprocessor(UUID moduleId){
		return getPreprocessors().get(moduleId);
	}
	
	/**
	 * Get Memory modules for this neural network instance
	 * @return
	 */
	Map<UUID, Memory> getMemories();
	
	/**
	 * Get Memory module by id
	 * @param moduleId
	 * @return
	 */
	default Memory getMemory(UUID moduleId){
		return getMemories().get(moduleId);
	}
	
	/**
	 * Get all modules for this neural network instance
	 * @return
	 */
	Map<UUID, Module> getModules();
	
	/**
	 * Get Module by id
	 * @param moduleId
	 * @return
	 */
	default Module getModule(UUID moduleId){
		return getModules().get(moduleId);
	}
	
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
	 * Fill the parameters with random values
	 */
	void randomizeParameters();

	/**
	 * Fill the parameters of a certain module with random values
	 */
	void randomizeParameters(UUID moduleId);
	
	/**
	 * Store the current parameters into the repository
	 * @param tag
	 */
	void storeParameters(String... tag);

	/**
	 * Store the current parameters into the repository
	 * @param tag
	 */
	void storeParameters(Map<UUID, Tensor> params, String... tag);
	
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
	
	/**
	 * Reset deltaparameters to zero
	 */
	void zeroDeltaParameters();
	
	/**
	 * Add deltaparameters to the parameters
	 */
	void updateParameters();
	
	/**
	 * Accumulate the gradients in deltaprameters
	 */
	void accGradParameters();
	
	/**
	 * Reset memory
	 */
	void resetMemory(int batchSize);
	
	default void resetMemory(){
		resetMemory(0);
	}
}
