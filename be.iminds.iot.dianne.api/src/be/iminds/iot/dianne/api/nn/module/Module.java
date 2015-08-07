package be.iminds.iot.dianne.api.nn.module;

import java.util.EnumSet;
import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Module is the basic building block of neural networks in Dianne. It provides
 * two flows of information. A forward pass in which input data is transformed in some way
 * to give an output, and a backward pass that takes in the gradient on the output of the 
 * previous forward pass and expects the corresponding gradient on the input. 
 * 
 * Each Module can have one (or more) next Modules to forward its output to, and one (or more)
 * Modules to propagate the gradient on the input to in the backward pass. A neural network
 * can be constructed by chaining a number of Modules. With each forward and backward, a number 
 * of String tags can optionally be provided, this in order to be able to tag an input at the one 
 * end of the neural network chain and identify the output at the other end. 
 * 
 * Modules can also have ForwardListeners and BackwardListeners that are notified when this
 * Module has done a forward resp. backward pass.
 * 
 * @author tverbele
 *
 */
public interface Module {
	
	public enum Mode {BLOCKING, SKIP, FORWARD_ON_CHANGE, WAIT_FOR_ALL}
	
	/**
	 * Get the identifier of this Module. This identifies the Module type and configuration, but
	 * not the instance. In fact, there can be multiple Module instances with the same Module id
	 * active at the same time.
	 *  
	 * @return the UUID of this Module
	 */
	UUID getId();

	/**
	 * Perform a forward pass
	 * @param moduleId the UUID of the Module where this input comes from
	 * @param input the actual input data
	 * @param tags optional tags for identifying this input
	 */
	void forward(final UUID moduleId, final Tensor input, final String... tags);
	
	/**
	 * Perform a backward pass
	 * @param moduleId the UUID of the Module where this gradOutput comes from
	 * @param gradOutput the actual gradient data on the previous output
	 * @param tags optional tags for identifying this gradOutput
	 */
	void backward(final UUID moduleId, final Tensor gradOutput, final String... tags);
	
	/**
	 * Configure the next Module(s)
	 * @param next array of Modules to call after a forward pass
	 */
	void setNext(final Module... next);
	
	/**
	 * Configure the previous Module(s)
	 * @param next array of Modules to call after a backward pass
	 */
	void setPrevious(final Module... prev);

	/**
	 * Add a listener to a forward pass
	 * @param listener forward listener to add
	 */
	void addForwardListener(ForwardListener listener);
	
	/**
	 * Remove a listener to a forward pass
	 * @param listener forward listener to remove
	 */
	void removeForwardListener(ForwardListener listener);
	
	/**
	 * Add a listener to a backward pass
	 * @param listener backward listener to add
	 */
	void addBackwardListener(BackwardListener listener);
	
	/**
	 * Remove a listener to a backward pass
	 * @param listener backward listener to remove
	 */
	void removeBackwardListener(BackwardListener listener);
	
	/**
	 * Set the working mode of this Module, possible options are:
	 *  BLOCKING : block until next is done
	 *  SKIP : in case next is still busy, skip this frame
	 *  FORWARD_ON_CHANGE : in case of Fork/Join: forward each time a subset of the input changes
	 *  WAIT_FOR_ALL : in case of Fork/Join : wait for all inputs to be gathered before forwarding
	 * 
	 * @param mode the mode to set
	 */
	void setMode(EnumSet<Mode> mode);
}
