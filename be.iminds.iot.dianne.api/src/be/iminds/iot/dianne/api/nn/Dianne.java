package be.iminds.iot.dianne.api.nn;

import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * Convenient service to create NeuralNetwork services from instance dto
 * 
 * @author tverbele
 *
 */
public interface Dianne {

	Promise<NeuralNetwork> getNeuralNetwork(NeuralNetworkInstanceDTO nn);
	
}