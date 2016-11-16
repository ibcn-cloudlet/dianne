package be.iminds.iot.dianne.api.rl.agent;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Interface used by manual action strategy allowing to provide expert input in an environment
 * 
 * @author tverbele
 *
 */
public interface ActionController {

	void setAction(Tensor action);
	
}
